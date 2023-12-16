import streamlit as st
import optuna
from keras.models import Sequential
from keras.layers import Embedding, LSTM, Dense
from keras.callbacks import EarlyStopping


def objective(trial):
    # Hyperparameter search space
    embedding_dim = trial.suggest_int('embedding_dim', 50, 300)
    lstm_units = trial.suggest_int('lstm_units', 50, 300)
    batch_size = trial.suggest_categorical('batch_size', [32, 64, 128])
    epochs = trial.suggest_int('epochs', 5, 50)

    # Build the model
    model = Sequential()
    model.add(Embedding(len(word_model.wv.vocab)+1, embedding_dim, input_length=X.shape[1], weights=[embedding_matrix], trainable=False))
    model.add(LSTM(lstm_units, return_sequences=False))
    model.add(Dense(y.shape[1], activation="softmax"))

    # Compile the model
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=['acc'])

    # Early stopping to prevent overfitting
    early_stopping = EarlyStopping(monitor='val_loss', patience=5)

    # Train the model
    model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs, validation_data=(X_val, y_val), callbacks=[early_stopping])

    # Evaluate the model on the validation set
    val_loss, val_acc = model.evaluate(X_val, y_val)

    return val_loss

def main():
    st.title("Optuna + Streamlit Example")

    # Create a study object and optimize
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=50)

    # Get the best hyperparameters
    best_params = study.best_params
    best_embedding_dim = best_params['embedding_dim']
    best_lstm_units = best_params['lstm_units']
    best_batch_size = best_params['batch_size']
    best_epochs = best_params['epochs']

    # Display the best hyperparameters
    st.subheader("Best Hyperparameters")
    st.write(f"Embedding Dimension: {best_embedding_dim}")
    st.write(f"LSTM Units: {best_lstm_units}")
    st.write(f"Batch Size: {best_batch_size}")
    st.write(f"Epochs: {best_epochs}")

    # Train the final model with the best hyperparameters
    final_model = Sequential()
    final_model.add(Embedding(len(word_model.wv.vocab)+1, best_embedding_dim, input_length=X.shape[1], weights=[embedding_matrix], trainable=False))
    final_model.add(LSTM(best_lstm_units, return_sequences=False))
    final_model.add(Dense(y.shape[1], activation="softmax"))

    final_model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=['acc'])
    final_model.fit(X_train, y_train, batch_size=best_batch_size, epochs=best_epochs)