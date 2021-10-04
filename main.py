import numpy as np
import pandas as pd

from pathlib import Path

import QuestionAnswers.part2 as part2

pd.options.mode.chained_assignment = None  # default='warn'


def main():
    # part2.q2a()
    # part2.q2b()
    # part2.q2d()
    part2.q4a()
    part2.q4b() # Definitely ain't workin . . .

    # import json
    # print('final tree: ', json.dumps(model.tree, sort_keys=True, indent=2))

main()