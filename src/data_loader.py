import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

def load_and_split_data(config):
    df = pd.read_csv(config.DATA_PATH)
    
    # 7:2 분할 (Group 기반)
    gss = GroupShuffleSplit(n_splits=1, test_size=config.VAL_SIZE, random_state=config.RANDOM_STATE)
    train_idx, val_idx = next(gss.split(df, groups=df[config.GROUP_COL]))
    
    train_df = df.iloc[train_idx]
    val_df = df.iloc[val_idx]
    
    train_files = sorted(train_df[config.GROUP_COL].unique().tolist())
    val_files = sorted(val_df[config.GROUP_COL].unique().tolist())

    print(f"\n📦 [Data Split: 7(Train) vs 2(Val)]")
    print(f" 🟢 Train ({len(train_files)} files): {train_files}")
    print(f" 🟡 Val   ({len(val_files)} files): {val_files}")
    print("-" * 50)

    def get_x_y(data):
        X = data.drop(columns=[config.TARGET_COL, config.GROUP_COL])
        y = data[config.TARGET_COL]
        return X, y

    X_train, y_train = get_x_y(train_df)
    X_val, y_val = get_x_y(val_df)
    
    return X_train, y_train, X_val, y_val