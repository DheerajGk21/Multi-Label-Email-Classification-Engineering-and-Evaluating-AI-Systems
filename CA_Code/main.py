from preprocess import *
from embeddings import *
from modelling.modelling import *
from modelling.data_model import *
import random

seed = 0
random.seed(seed)
np.random.seed(seed)


def load_data():
    # load the input data
    df = get_input_data()
    return df


def preprocess_data(df):
    # De-duplicate input data
    df = de_duplication(df)
    # remove noise in input data
    df = noise_remover(df)
    # translate data to english
    # df[Config.TICKET_SUMMARY] = translate_to_en(df[Config.TICKET_SUMMARY].tolist())
    return df


def get_embeddings(df: pd.DataFrame):
    X = get_tfidf_embd(df)  # get tf-idf embeddings
    return X, df


def get_data_object(X: np.ndarray, df: pd.DataFrame, label: str):
    return Data(X, df, label)


def perform_modelling(data: Data, df: pd.DataFrame, name):
    return model_predict(data, df, name)



if __name__ == "__main__":
    df = load_data()
    df = preprocess_data(df)
    df[Config.INTERACTION_CONTENT] = df[Config.INTERACTION_CONTENT].values.astype("U")
    df[Config.TICKET_SUMMARY] = df[Config.TICKET_SUMMARY].values.astype("U")
    grouped_df = df.groupby(Config.GROUPED)
    for name, group_df in grouped_df:
        print("=====================================")
        print("Group name", name)

        # ----------- My Code ------------------------
        X_current, group_df = get_embeddings(group_df)
        group_df_current = group_df.copy()  # Copy to keep original dataframe intact
        predictions = {}  # Store model outputs

        # Chain the models
        for label in Config.LABEL_CHAIN:
            print("-------------------------------------")
            print("\n Current Model label : ", label)

            data_obj = get_data_object(X_current, group_df_current, label)

            if data_obj.X_train is None:
                print("Model " + label + "skipped due to insufficient data.")
                continue

            # Train the model and predict the outcome
            y_pred = perform_modelling(data_obj, group_df_current, label)
            predictions[label] = y_pred  # Store the output for chaining

            # Add predictions as new feature for next stage
            y_pred_column = np.array(y_pred).reshape(-1, 1)
            X_current = combine_embd(X_current, y_pred_column)

            group_df_current[label + "_pred"] = (
                y_pred.flatten()
            )  # Also add the the predictions just for analysis
            group_df_current[label + "_true"] = data_obj.get_type_encoded_series()

        # Print Results
        print("-------------------------------------")
        print("\n Model " + name + " Output :")
        print("-------------------------------------")
        for label, preds in predictions.items():
            print(label + ": ", np.unique(preds, return_counts=True))
