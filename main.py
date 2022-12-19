model_ckpt = "model.ckpt"
if joint: 
    ...
else:
    model = basic_twostep_model()

CalcMetrics(model, test_dataloader)

