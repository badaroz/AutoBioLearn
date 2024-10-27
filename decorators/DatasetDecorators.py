def requires_dataset(func):
    def wrapper(self, *args, **kwargs):
        dataset = getattr(self, 'dataset', None) or getattr(getattr(self, 'data_processor', None), 'dataset', None)        
        if dataset is None:
            raise ValueError("Dataset is not initialized.")
        return func(self, *args, **kwargs)
    return wrapper
