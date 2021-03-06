import types
import tempfile
import keras.models

def make_keras_picklable():
    """
    In this project the different datasets and models are saved into pickle files, 
    that involves saving Keras models.
    Unfortunately, there is a known problem when trying to save Keras models into pickle files, 
    a workaround can be found on http://zachmoshe.com/2017/04/03/pickling-keras-models.html, 
    which code is included bellow and used for the purposes of this project.
    """
    def __getstate__(self):
        model_str = ""
        with tempfile.NamedTemporaryFile(suffix='.hdf5', delete=True) as fd:
            keras.models.save_model(self, fd.name, overwrite=True)
            model_str = fd.read()
        d = { 'model_str': model_str }
        return d

    def __setstate__(self, state):
        with tempfile.NamedTemporaryFile(suffix='.hdf5', delete=True) as fd:
            fd.write(state['model_str'])
            fd.flush()
            model = keras.models.load_model(fd.name)
        self.__dict__ = model.__dict__


    cls = keras.models.Model
    cls.__getstate__ = __getstate__
    cls.__setstate__ = __setstate__

