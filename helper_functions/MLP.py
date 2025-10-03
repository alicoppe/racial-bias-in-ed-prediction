from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout, Concatenate
from tensorflow.keras.optimizers.legacy import Adam

class MLP():
    def __init__(self, layer1_values, layer2_values, shape_text, shape_num, dropout=0):
        self.layer1_values = layer1_values
        self.layer2_values = layer2_values
        
        self.shape_text = shape_text
        self.shape_num = shape_num
        
        self.dropout = dropout
        self.optimizer = 'adam'
        self.model = self._create_nn()
        
        self.train_loss = []
        self.val_loss = [] 

    def _create_nn(self):
        text_input = Input(shape=(self.shape_text,), name='encoded_text_input')
        combined_input = Input(shape=(self.shape_num,), name='numeric_input')

        text_model = text_input
        for value in self.layer1_values:
            text_model = Dense(value, activation='relu')(text_model)
            if self.dropout > 0:
                text_model = Dropout(self.dropout)(text_model)

        combined_with_text = Concatenate()([combined_input, text_model])

        for value in self.layer2_values:
            combined_with_text = Dense(value, activation='relu')(combined_with_text)
            if self.dropout > 0:
                combined_with_text = Dropout(self.dropout)(combined_with_text)

        output = Dense(1, activation='sigmoid')(combined_with_text)

        return Model(inputs=[combined_input, text_input], outputs=output)

    def train(self, X_data, Y_data, loss='binary_crossentropy', callback=None, batch_size=32, epochs=10, verbose=1, validation_data=None):
        
        self.optimizer.build(self.model.trainable_variables)
        self.model.compile(optimizer=self.optimizer, loss=loss, metrics=['accuracy'])

        fit_params = {
            'x': X_data,
            'y': Y_data,
            'epochs': epochs,
            'batch_size': batch_size,
            'verbose': verbose,
        }

        if callback:
            fit_params['callbacks'] = [callback]

        if validation_data:
            fit_params['validation_data'] = validation_data

        history = self.model.fit(**fit_params)

        self.train_loss = history.history['loss']
        self.val_loss = history.history.get('val_loss', [])
        
        return

    def predict(self, X_data):
        return self.model.predict(X_data)
    
    
## For some reason, there is a problem with the optimizer. It doesn't seem to be working. Will need to revise this and possibly implement in the future.
# As of now just using the function calls instead, but they're not compatible with returning the validation data.

# def run_predictions(X_train, X_test, Y_train, Y_test, verbose=1, epochs=3, batch_size=32, train_set=True, val_split=False):
#     Y_train = Y_train.astype('float32')
#     Y_test = Y_test.astype('float32')
    
#     X_train_num, X_test_num, X_train_encoded, X_test_encoded = encode_and_split_new(X_train, X_test)
    
#     if val_split:
#         concatenated = np.column_stack((X_train_num, X_train_encoded, Y_train))

#         # Train test split
#         train, test = train_test_split(concatenated, test_size=0.2, random_state=42)

#         # Split train and test sets by the original columns
#         X_train_num = train[:, :X_train_num.shape[1]]
#         X_train_encoded = train[:, X_train_num.shape[1]:-1]
#         Y_train = train[:, -1]

#         X_val_num = test[:, :X_train_num.shape[1]]
#         X_val_encoded = test[:, X_train_num.shape[1]:-1]
#         Y_val = test[:, -1]
        
#         validation_data = ([X_val_num, X_val_encoded], Y_val)
        
#     else:
#         validation_data = None
    
#     metrics_callback = MetricsCallback(X_test_num, X_test_encoded, Y_test)
    
#     model = MLP([64, 32], [32], X_train_encoded.shape[1], X_train_num.shape[1])
#     model._create_nn()
#     model.train([X_train_num, X_train_encoded], Y_train, 
#                 loss=binary_expected_calibration_error,
#                 callback=metrics_callback, 
#                 batch_size=batch_size,
#                 epochs=epochs,
#                 verbose=verbose,
#                 validation_data=validation_data)

    
#     if train_set:
#         return model.predict([X_train_num, X_train_encoded]), model.train_loss, model.val_loss
#     else:
#         return model.predict([X_test_num, X_test_encoded]), model.train_loss, model.val_loss
