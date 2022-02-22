import pandas as pd
import os
from glob import glob


def dataframe_from_dir(input_dir="input/aug"):
    # get cell-types by sub-directory names
    cells = [c.split(os.sep)[-1] for c in glob(f'{input_dir}/*')]

    # build df
    df = pd.DataFrame(columns=["cell_type", "path"])
    i  = 0

    for c in cells:
        for img_path in glob(f'{input_dir}/{c}/*.jpg'):
            df.loc[i] = [c, img_path]
            i += 1

    # map cell name to number
    df['y'] = pd.factorize(df['cell_type'], sort=True)[0]
    return df