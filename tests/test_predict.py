import pytest
from pathlib import Path

from src.schemas.state import load_image_model
from src.services.predict import predict_image

TEST_IMAGE = Path(__file__).parent / 'test_clothing.jpg'

@pytest.fixture(scope='module')
def image_model():
    # Loading the model is slow so share it across the tests in this file
    return load_image_model()

def test_image_output(image_model):
    preds = predict_image(image_model, TEST_IMAGE.read_bytes())
    assert preds['color'] in image_model.color_names
    assert preds['category'] in image_model.category_names
    assert all(a in image_model.apparels for a in preds['attributes'])


'''
def test_get_user_params():
    assert get_user_params()

def test_initialize_preds():
    assert initialize_preds()

def test_query_rag_system():
    assert query_rag_system()
'''
