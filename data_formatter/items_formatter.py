import pandas as pd
import re

class ItemFormatter():

    def __init__(self, items_df: pd.DataFrame) -> None:
        print(f"{len(items_df.index)} initial items.")
        self.items_df = items_df.copy()
        self.missing_desc_ids = list(self.items_df[self.items_df.description.isna()]['itemId'])
        self.items_df.dropna(inplace=True)

    def clean_spaces(self, text):
        return re.sub(r'(?<=[.,])(?=[^\s])', r' ', re.sub(r"\s{2,}", " ", re.sub('[\W+\s[.]]',' ', text))) 
    
    def process(self):
        self.items_df.loc[:,"description"] = self.items_df.description.apply(self.clean_spaces).str.replace('""', "").replace("\t"," ")
        self.items_df.set_index("itemId", inplace=True)
        return self.items_df
    