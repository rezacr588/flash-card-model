from joblib import dump, load

def save_model(model, filename='src/trained_model.joblib'):
    dump(model, filename)

def load_model(filename='src/trained_model.joblib'):
    return load(filename)
