import argparse

import pandas as pd
import wikipedia
import yaml

from utils import set_seed


def create_alias(df):
    listAns = list(df.Answer_text)[:]

    aliases = []
    for i, ans in enumerate(listAns):
        if (i + 1) % 50 == 0:
            print(f"{i+1} samples done")
        if ans == "[]":
            aliases.append(["", "", ""])
        else:
            try:
                result = wikipedia.search(ans[2:-2], results=3)
                for count in range(len(result)):
                    result[count] = result[count].replace("'", "")
                while len(result) < 3:
                    result.append("")
                aliases.append(result)
            except:
                print(ans)
                print(i)
                raise NotImplementedError

        dfx2 = df.append(df[:])

        dfx4 = dfx2.append(dfx2[:])

        listStart = list(dfx4.Answer_start)[:]

        num = len(listAns)

        for i in range(3 * num):
            listAns.append(aliases[i % num][i // num])
            if listAns[num + i] == "":
                listAns[num + i] = "[]"
                listStart[num + i] = "[]"
            else:
                listAns[num + i] = "['" + listAns[num + i] + "']"

        dfx4.Answer_text = listAns
        dfx4.Answer_start = listStart

        return dfx4
