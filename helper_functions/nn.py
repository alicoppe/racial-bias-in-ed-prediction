from tensorflow.keras.layers import Input, Dense, Concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dropout
from tensorflow.keras.initializers import GlorotUniform

def create_nn(layer1_values, layer2_values, shape_text, shape_num, seed=42, dropout=0):
    
    initializer = GlorotUniform(seed=seed)

    # Define input shapes
    text_input = Input(shape=(shape_text,), name='encoded_text_input')
    combined_input = Input(shape=(shape_num,), name='numeric_input')

    # Define neural network for text data with optional dropout
    text_model = text_input
    for units in layer1_values:
        text_model = Dense(units, activation='relu', kernel_initializer=initializer)(text_model)
        if dropout > 0:
            text_model = Dropout(dropout)(text_model)

    # Concatenate text model output with combined numerical/categorical input
    combined_with_text = Concatenate()([combined_input, text_model])

    # Define additional layers if needed with optional dropout
    for units in layer2_values:
        combined_with_text = Dense(units, activation='relu', kernel_initializer=initializer)(combined_with_text)
        if dropout > 0:
            combined_with_text = Dropout(dropout)(combined_with_text)

    # Output layer
    output = Dense(1, activation='sigmoid', kernel_initializer=initializer)(combined_with_text)

    # Define model
    model = Model(inputs=[combined_input, text_input], outputs=output)
    
    return model
