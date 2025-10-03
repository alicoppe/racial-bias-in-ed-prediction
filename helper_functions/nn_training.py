from keras.optimizers import Adam

def train_model(model, X_data, Y_data, 
                opt=Adam(learning_rate=0.001, decay=1e-6), 
                loss='binary_crossentropy', 
                callback=None, 
                batch_size=32,
                epochs=10,
                verbose=1,
                validation_data=None):
    
    model.compile(optimizer=opt, 
                  loss=loss, 
                  metrics=['accuracy'])

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

    history = model.fit(**fit_params)
    
    # Extract loss metrics
    train_loss = history.history['loss']
    val_loss = history.history['val_loss'] if 'val_loss' in history.history else []
    
    return model