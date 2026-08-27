import pytest
from polyattn import masks


@pytest.fixture(params=masks.zoo(), ids=lambda m: m.name)
def mask(request):
    return request.param
