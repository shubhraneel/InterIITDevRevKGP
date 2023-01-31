import json
import os

import sys
import time

import numpy as np

import onnxruntime
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
from data import SQuAD_Dataset, SQuAD_TestDataset
from onnxruntime.quantization import quantize_dynamic
from sklearn.metrics import accuracy_score, classification_report, f1_score
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import get_scheduler

from transformers.modeling_outputs import QuestionAnsweringModelOutput

from utils import compute_f1
from itertools import accumulate

def to_numpy(tensor):
    return (
        tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()
    )


class Trainer:
    def __init__(
        self,
        config,
        model,
        optimizer,
        device,
        tokenizer,
        ques2idx,
        title2idx,
        df_val=None,
        val_retriever=None,
    ):
        self.config = config
        self.device = device

        self.tokenizer = tokenizer

        self.optimizer = optimizer
        self.model = model

        # wandb.watch(self.model)

        self.ques2idx = ques2idx
        self.title2idx = title2idx

        self.df_val = df_val
        self.val_retriever = val_retriever

        self.prepared_test_loader = None
        self.prepared_test_df_matched = None

        if self.config.training.lr_flag:
            self.scheduler = get_scheduler(
                self.config.scheduler,
                self.optimizer,
                num_warmup_steps=0,
                num_training_steps=1840 * self.config.training.epochs,
            )

        self.best_val_loss = 1e9

        # setup onnx runtime if config.onnx is true
        # self.onnx_runtime_session = None
        # if self.config.ONNX:
        #     self.model.export_to_onnx(tokenizer)

        #     # TODO Handle this case when using quantization without ONNX using torch.quantization

        #     if self.config.quantize:
        #         quantize_dynamic(
        #             "checkpoints/{}/model.onnx".format(self.config.load_path),
        #             "checkpoints/{}/model_quantized.onnx".format(self.config.load_path),
        #         )

        #     sess_options = onnxruntime.SessionOptions()
        #     # TODO Find if this line helps
        #     # sess_options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
        #     model_path = (
        #         "checkpoints/{}/model.onnx".format(self.config.load_path)
        #         if not self.config.quantize
        #         else "checkpoints/{}/model_quantized.onnx".format(self.config.load_path)
        #     )
        #     self.onnx_runtime_session = onnxruntime.InferenceSession(
        #         model_path, sess_options
        #     )

        if self.config.model.span_level:
            seq_indices = list(range(self.config.data.max_length))
            self.seq_pair_indices = [
                (x, y)
                for x in seq_indices
                for y in seq_indices
                if y - x >= 0 and y - x <= config.data.answer_max_len
            ]

    def _train_step(self, dataloader, epoch):
        total_loss = 0
        tepoch = tqdm(dataloader, unit="batch", position=0, leave=True)

        for batch_idx, batch in enumerate(tepoch):
            tepoch.set_description(f"Epoch {epoch + 1}")
            if len(batch["question_context_input_ids"].shape) == 1:
                batch["question_context_input_ids"] = batch[
                    "question_context_input_ids"
                ].unsqueeze(dim=0)
                batch["question_context_attention_mask"] = batch[
                    "question_context_attention_mask"
                ].unsqueeze(dim=0)
                if not self.config.model.non_pooler:
                    batch["question_context_token_type_ids"] = batch[
                        "question_context_token_type_ids"
                    ].unsqueeze(dim=0)

            out = self.model(batch)
            if batch_idx % 300 == 0:
                self.log_ipop_batch(batch, out, batch_idx)

            if self.config.model.span_level:
                loss = out[0].loss
            else:
                loss = out.loss
            loss.backward()

            total_loss += loss.item()
            tepoch.set_postfix(loss=total_loss / (batch_idx + 1))
            wandb.log({"train_batch_loss": total_loss / (batch_idx + 1)})

            self.optimizer.step()
            if self.config.training.lr_flag:
                self.scheduler.step()
            self.optimizer.zero_grad()

        wandb.log({"train_epoch_loss": total_loss / (batch_idx + 1)})

        return total_loss / (batch_idx + 1)

    def log_ipop_batch(self, batch, out, batch_idx):
        rows = []
        for i in range(len(batch["context"])):
            context = batch["context"][i]

            if self.config.model.span_level:
                probs = torch.sigmoid(out[1])
                max_probs = torch.max(probs, axis=1)

                idx = max_probs.indices[i].item()
                start_index, end_index = self.seq_pair_indices[idx]
            else:
                start_probs = F.softmax(out.start_logits, dim=1)
                end_probs = F.softmax(out.end_logits, dim=1)

                max_start_probs = torch.max(start_probs, axis=1)
                max_end_probs = torch.max(end_probs, axis=1)

                start_index = max_start_probs.indices[i].item()
                end_index = max_end_probs.indices[i].item()

            offset_mapping = batch["question_context_offset_mapping"][i]
            decoded_answer = ""
            if (
                offset_mapping[start_index] is not None
                and offset_mapping[end_index] is not None
            ):
                start_char = offset_mapping[start_index][0]
                end_char = offset_mapping[end_index][1]

                decoded_answer = context[start_char:end_char]

            tgt_answer = ""
            if (
                offset_mapping[batch["start_positions"][i]] is not None
                and offset_mapping[batch["end_positions"][i]] is not None
            ):
                tgt_start = offset_mapping[batch["start_positions"][i]][0]
                tgt_end = offset_mapping[batch["end_positions"][i]][1]

                tgt_answer = context[tgt_start:tgt_end]

            answer = batch["answer"][i]
            question = batch["question"][i]

            rows.append([question, answer, decoded_answer, tgt_answer])

        my_table = wandb.Table(
            columns=["question", "dataset answer", "predicted answer", "train target"],
            data=rows,
        )
        wandb.log({"Batch " + str(batch_idx) + " IP/OP": my_table})

    def train(self, train_dataloader, val_dataloader=None):
        self.model.train()
        for epoch in range(self.config.training.epochs):
            self._train_step(train_dataloader, epoch)

            if (val_dataloader is not None) and (
                ((epoch + 1) % self.config.training.evaluate_every)
            ) == 0:
                val_loss = self.evaluate(val_dataloader)
                if epoch == 0:
                    metrics = self.calculate_metrics(
                        self.df_val,
                        self.val_retriever,
                        "val",
                        self.device,
                        do_prepare=True,
                    )
                else:
                    metrics = self.calculate_metrics(
                        self.df_val,
                        self.val_retriever,
                        "val",
                        self.device,
                        do_prepare=False,
                    )
                if self.best_val_loss >= val_loss and self.config.save_model_optimizer:
                    self.best_val_loss = val_loss
                    print(
                        "saving best model and optimizer at checkpoints/{}/model_optimizer.pt".format(
                            self.config.load_path
                        )
                    )
                    os.makedirs(
                        "checkpoints/{}/".format(self.config.load_path), exist_ok=True
                    )
                    torch.save(
                        {
                            "model_state_dict": self.model.state_dict(),
                            "optimizer_state_dict": self.optimizer.state_dict(),
                        },
                        "checkpoints/{}/model_optimizer.pt".format(
                            self.config.load_path
                        ),
                    )
                self.model.train()

    def evaluate(self, dataloader):
        self.model.eval()
        total_loss = 0
        tepoch = tqdm(dataloader, unit="batch", position=0, leave=True)
        tepoch.set_description("Validation Step")
        with torch.no_grad():
            for batch_idx, batch in enumerate(tepoch):
                if len(batch["question_context_input_ids"].shape) == 1:
                    batch["question_context_input_ids"] = batch[
                        "question_context_input_ids"
                    ].unsqueeze(dim=0)
                    batch["question_context_attention_mask"] = batch[
                        "question_context_attention_mask"
                    ].unsqueeze(dim=0)
                    if not self.config.model.non_pooler:
                        batch["question_context_token_type_ids"] = batch[
                            "question_context_token_type_ids"
                        ].unsqueeze(dim=0)

                out = self.model(batch)
                if self.config.model.span_level:
                    loss = out[0].loss
                else:
                    loss = out.loss

                total_loss += loss.item()
                tepoch.set_postfix(loss=total_loss / (batch_idx + 1))
                wandb.log({"val_batch_loss": total_loss / (batch_idx + 1)})

        wandb.log({"val_epoch_loss": total_loss / (batch_idx + 1)})
        return total_loss / (batch_idx + 1)

    def predict(self, batch, onnx_session):
        if self.config.ONNX:
            # Set up inputs for the ONNX Runtime Invocation
            ort_inputs = None
            if not self.config.model.non_pooler:
                ort_inputs = {
                    onnx_session.get_inputs()[0].name: to_numpy(
                        batch["question_context_input_ids"]
                    ),
                    onnx_session.get_inputs()[1].name: to_numpy(
                        batch["question_context_attention_mask"]
                    ),
                    onnx_session.get_inputs()[2].name: to_numpy(
                        batch["question_context_token_type_ids"]
                    ),
                    # onnx_session.get_inputs()[3].name: to_numpy(batch['start_positions'].unsqueeze(dim=1)),
                    # onnx_session.get_inputs()[4].name: to_numpy(batch['end_positions'].unsqueeze(dim=1))
                }

            else:
                ort_inputs = {
                    onnx_session.get_inputs()[0].name: to_numpy(
                        batch["question_context_input_ids"]
                    ),
                    onnx_session.get_inputs()[1].name: to_numpy(
                        batch["question_context_attention_mask"]
                    ),
                    # onnx_session.get_inputs()[2].name: to_numpy(batch['start_positions'].unsqueeze(dim=1)),
                    # onnx_session.get_inputs()[3].name: to_numpy(batch['end_positions'].unsqueeze(dim=1))
                }

            # print(ort_inputs)

            ort_outputs = onnx_session.run(None, ort_inputs)

            # print(ort_outputs)
            out = QuestionAnsweringModelOutput(
                # loss = torch.tensor(ort_outputs[0]),
                start_logits=torch.tensor(ort_outputs[0]),
                end_logits=torch.tensor(ort_outputs[1]),
            )

            return out

        return self.model(batch)

    def prepare_df_before_inference(self, df_test, retriever, prefix, device):
        title_id_list = df_test["title_id"].unique()
        gb_title = df_test.groupby("title_id")
        df_test_matched = pd.DataFrame()

        df_unique_con = df_test.drop_duplicates(subset=["context"])
        unmatched = 0

        start_time = time.time()
        for title_id in title_id_list:
            df_temp = gb_title.get_group(title_id)
            # can initialise theme specific retriever here
            for idx, row in df_temp.iterrows():
                df_contexts = pd.DataFrame()
                if retriever is not None and self.config.sentence_level:
                    question = row["question"]
                    question_id = row["question_id"]
                    context_id = row["context_id"]
                    doc_idx_filtered, doc_text_filtered = retriever.retrieve_top_k(
                        question, str(title_id), k=self.config.top_k
                    )
                    df_contexts = df_unique_con.loc[
                        df_unique_con["title_id"] == title_id
                    ].sample(n=1, random_state=self.config.seed)
                    df_contexts.loc[:, "question"] = question
                    df_contexts.loc[:, "question_id"] = question_id
                    df_contexts.loc[:, "answerable"] = False
                    df_contexts.loc[:, "answer_start"] = ""
                    df_contexts.loc[:, "answer_text"] = ""
                    df_contexts.loc[:, "context"] = "".join(doc_text_filtered)
                    df_contexts.loc[:, "context_id"] = "+".join(doc_text_filtered)
                else:
                    question = row["question"]
                    question_id = row["question_id"]
                    context_id = row["context_id"]

                    if retriever is not None:
                        doc_idx_filtered, doc_text_filtered = retriever.retrieve_top_k(
                            question, str(title_id), k=self.config.top_k
                        )
                        df_contexts_og = df_unique_con.loc[
                            df_unique_con["context_id"].isin(
                                [int(doc_idx) for doc_idx in doc_idx_filtered]
                            )
                        ].copy()
                        # TODO: we can endup sampling things in doc_idx_filtered again
                        df_contexts_random = df_unique_con.loc[
                            df_unique_con["title_id"] == title_id
                        ].sample(
                            n=max(0, self.config.top_k - len(doc_idx_filtered)),
                            random_state=self.config.seed,
                        )
                        df_contexts = pd.concat(
                            [df_contexts_og, df_contexts_random],
                            axis=0,
                            ignore_index=True,
                        )
                    else:
                        df_contexts = df_unique_con.loc[
                            df_unique_con["title_id"] == title_id
                        ].sample(n=self.config.top_k, random_state=self.config.seed)
                    df_contexts.loc[:, "question"] = question
                    df_contexts.loc[:, "question_id"] = question_id
                    df_contexts.loc[:, "answerable"] = False
                    df_contexts.loc[:, "answer_start"] = ""
                    df_contexts.loc[:, "answer_text"] = ""

                    original_context_idx = df_contexts.loc[
                        df_contexts["context_id"] == context_id
                    ]
                    if len(original_context_idx) == 0:
                        # print(f"original paragraph not in top k {unmatched}")
                        unmatched += 1
                    else:
                        row_dict = row.to_dict()
                        df_contexts.loc[
                            df_contexts["context_id"] == context_id, row_dict.keys()
                        ] = row_dict.values()
                df_test_matched = pd.concat(
                    [df_test_matched, df_contexts], axis=0, ignore_index=True
                )

        # print(f"original paragraph not in top k {unmatched}")

        test_ds = SQuAD_Dataset(
            self.config, df_test_matched, self.tokenizer
        )  # , hide_tqdm=True
        test_dataloader = DataLoader(
            test_ds,
            batch_size=self.config.data.val_batch_size,
            collate_fn=test_ds.collate_fn,
        )
        time_test_dataloader_generation = 1000 * (time.time() - start_time)
        print(time_test_dataloader_generation)
        print(time_test_dataloader_generation / df_test.shape[0])
        wandb.log(
            {
                "time_"
                + str(prefix)
                + "_dataloader_generation": time_test_dataloader_generation
            }
        )
        wandb.log(
            {
                "per_q_"
                + str(prefix)
                + "_time_"
                + str(prefix)
                + "_dataloader_generation": time_test_dataloader_generation
                / df_test.shape[0]
            }
        )

        print(f"{len(df_test_matched)=}")
        print(f"{len(df_temp)=}")

        self.prepared_test_loader = test_dataloader
        self.prepared_test_df_matched = df_test_matched

    def prepare_theme_df_before_inference(self, theme, theme_questions, theme_contexts, retriever, prefix, device):

        df_test_matched = pd.DataFrame()

        df_unique_con = theme_contexts

        df_temp = theme_questions

        start_time = time.time()

        title_id = str(self.title2idx[theme])

        for idx, row in df_temp.iterrows():
            df_contexts = pd.DataFrame()
            if retriever is not None and self.config.sentence_level:
                question = row["question"]
                question_id = row["question_id"]
                doc_idx_filtered, doc_text_filtered = retriever.retrieve_top_k(
                    question, title_id, k=self.config.top_k
                )
                new_row = dict()
                new_row['question'] = question
                new_row['question_id'] = question_id
                new_row['context'] = "".join(doc_text_filtered)
                new_row['context_id'] = "+".join(doc_idx_filtered)
                new_row['prefix_sum_lengths'] = [0]+list(accumulate([len(sent) for sent in doc_text_filtered]))
                print('sentence level row for question:', question)
                print(new_row)
                df_contexts = pd.DataFrame([new_row])
                
            else:
                question = row["question"]
                question_id = row["question_id"]

                if retriever is not None:
                    doc_idx_filtered, doc_text_filtered = retriever.retrieve_top_k(
                        question, title_id, k=self.config.top_k
                    )
                    df_contexts_og = df_unique_con.loc[
                        df_unique_con["context_id"].isin(
                            [int(doc_idx) for doc_idx in doc_idx_filtered]
                        )
                    ].copy()

                    # TODO: we can endup sampling things in doc_idx_filtered again
                    df_contexts_random = df_unique_con.sample(
                        n=max(0, self.config.top_k - len(doc_idx_filtered)),
                        random_state=self.config.seed,
                    )
                    df_contexts = pd.concat(
                        [df_contexts_og, df_contexts_random],
                        axis=0,
                        ignore_index=True,
                    )
                else:
                    df_contexts = df_unique_con.sample(n=self.config.top_k, random_state=self.config.seed)
                df_contexts["question"] = question
                df_contexts["question_id"] = question_id
            df_test_matched = pd.concat(
                [df_test_matched, df_contexts], axis=0, ignore_index=True
            )

        # print(f"original paragraph not in top k {unmatched}")

        df_test_matched['title'] = theme
        df_test_matched['title_id'] = self.title2idx[theme]

        test_ds = SQuAD_TestDataset(
            self.config, df_test_matched, self.tokenizer
        )  # , hide_tqdm=True
        test_dataloader = DataLoader(
            test_ds,
            batch_size=self.config.data.val_batch_size,
            collate_fn=test_ds.collate_fn,
        )
        time_test_dataloader_generation = 1000 * (time.time() - start_time)
        print(time_test_dataloader_generation)
        print(time_test_dataloader_generation / len(theme_questions))

        print(f"{len(df_test_matched)=}")
        print(f"{len(df_temp)=}")

        self.prepared_test_loader = test_dataloader
        self.prepared_test_df_matched = df_test_matched


    def inference(self, theme, questions, contexts, retriever, prefix, device, do_prepare, onnx_session):
        self.model.to(device)
        self.device = device
        self.model.device = device

        if do_prepare:
            self.prepare_theme_df_before_inference(theme, questions, contexts, retriever, prefix, device)
            print(self.prepared_test_df_matched)

        start_time = time.time()
        question_prediction_dict = {
            q_id: (0, "", -1)
            for q_id in self.prepared_test_df_matched["question_id"].unique()
        }

        # TODO: without sequentional batch iteration
        for qp_batch_id, qp_batch in tqdm(
            enumerate(self.prepared_test_loader), total=len(self.prepared_test_loader)
        ):
            if len(qp_batch["question_context_input_ids"].shape) == 1:
                qp_batch["question_context_input_ids"] = qp_batch[
                    "question_context_input_ids"
                ].unsqueeze(dim=0)
                qp_batch["question_context_attention_mask"] = qp_batch[
                    "question_context_attention_mask"
                ].unsqueeze(dim=0)
                if not self.config.model.non_pooler:
                    qp_batch["question_context_token_type_ids"] = qp_batch[
                        "question_context_token_type_ids"
                    ].unsqueeze(dim=0)

            # para, para_id, theme, theme_id, question, question_id
            pred = self.predict(qp_batch, onnx_session)

            # print(pred.start_logits.shape) -> [32,512]
            if self.config.model.span_level:
                probs = torch.sigmoid(pred[1])
                max_probs = torch.max(probs, axis=1)
                confidence_scores = max_probs.values
            else:
                start_probs = F.softmax(pred.start_logits, dim=1)  # -> [32,512]
                end_probs = F.softmax(pred.end_logits, dim=1)  # -> [32,512]

                max_start_probs = torch.max(start_probs, axis=1)  # -> [32,1]
                max_end_probs = torch.max(end_probs, axis=1)  # -> [32,1]

                confidence_scores = (
                    max_end_probs.values * max_start_probs.values
                )  # -> [32,1]

            for batch_idx, q_id in enumerate(qp_batch["question_id"]):
                if question_prediction_dict[q_id][0] < confidence_scores[batch_idx]:
                    # using the context in the qp_pair get extract the span using max_start_prob and max_end_prob
                    context = qp_batch["context"][batch_idx]
                    offset_mapping = qp_batch["question_context_offset_mapping"][
                        batch_idx
                    ]
                    decoded_answer = ""
                    pred_context_idx = -1

                    if self.config.model.span_level:
                        idx = max_probs.indices[batch_idx].item()
                        start_index, end_index = self.seq_pair_indices[idx]
                    else:
                        start_index = max_start_probs.indices[batch_idx].item()
                        end_index = max_end_probs.indices[batch_idx].item()

                    if (
                        offset_mapping[start_index] is not None
                        and offset_mapping[end_index] is not None
                    ):
                        start_char = offset_mapping[start_index][0]
                        end_char = offset_mapping[end_index][1]
                        decoded_answer = context[start_char:end_char]
                        prefix_sum_lengths = qp_batch['prefix_sum_lengths'][batch_idx]
                        for ret_idx, prefix_sum_length in enumerate(prefix_sum_lengths):
                            if start_char >= prefix_sum_length:
                                pred_context_idx=qp_batch[batch_idx]['context_id'].split('+')[ret_idx].split('_')[0]
                                if end_char> prefix_sum_length:
                                    sab_changa_si = False
                                    assert sab_changa_si, "maa chud gayi (end char bs)"
                                break
                    
                    if len(decoded_answer) > 0:
                        question_prediction_dict[q_id] = (
                            confidence_scores[batch_idx].item(),
                            decoded_answer,
                            pred_context_idx
                        )

        time_inference_generation = 1000 * (time.time() - start_time)
        print(time_inference_generation)
        print(time_inference_generation / len(questions))
        return question_prediction_dict

    def calculate_metrics(self, df_test, retriever, prefix, device, do_prepare):
        """
        1. Run the inference script
        2. Calculate the time taken
        3. Calculate the F1 score
        4. Return all
        """

        # TODO: check if this way of calculating the time is correct
        if device == "cuda":
            torch.cuda.synchronize()
        question_pred_dict = self.inference(
            df_test, retriever, prefix, device, do_prepare
        )
        predicted_answers = [
            question_pred_dict[q_id][1] for q_id in df_test["question_id"]
        ]
        gold_answers = df_test["answer_text"].tolist()

        assert len(predicted_answers) == len(gold_answers)
        squad_f1_per_span = [
            compute_f1(predicted_answers[i], gold_answers[i])
            for i in range(len(predicted_answers))
        ]
        mean_squad_f1 = np.mean(squad_f1_per_span)

        classification_prediction = [
            1 if (len(predicted_answers[i]) != 0) else 0
            for i in range(len(predicted_answers))
        ]
        classification_actual = df_test["answerable"].astype(int)
        classification_f1 = f1_score(classification_actual, classification_prediction)
        classification_accuracy = accuracy_score(
            classification_actual, classification_prediction
        )
        clf_report = classification_report(
            classification_actual, classification_prediction, output_dict=True
        )
        print(pd.DataFrame(clf_report).T)

        metrics = {
            "classification_accuracy": classification_accuracy,
            "clf_report": clf_report,
            "classification_f1": classification_f1,
            "mean_squad_f1": mean_squad_f1,
            # "mean_time_per_question (ms)": results["mean_time_per_question"]*1000,
        }

        if prefix == "val":
            wandb.log({"val_metrics": metrics})
            wandb.log({"val_predicted_answers": predicted_answers})
            wandb.log({"val_gold_answers": gold_answers})
        else:
            wandb.log({"metrics": metrics})
            wandb.log({"predicted_answers": predicted_answers})
            wandb.log({"gold_answers": gold_answers})

        return metrics
