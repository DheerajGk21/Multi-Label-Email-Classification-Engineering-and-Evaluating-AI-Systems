from model.randomforest import RandomForest


def model_predict(data, df, name):
    results = []
    print("RandomForest")
    model = RandomForest("RandomForest", data.get_embeddings(), data.get_type())
    model.train(data)
    model.predict(data.X_test)
    model.print_results(data)
    full_preds = model.mdl.predict(data.get_embeddings()) # Predict "Type 2/3/4" on entire dataset 
    return full_preds

def model_evaluate(model, data):
    model.print_results(data)