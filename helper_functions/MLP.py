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
    
    
