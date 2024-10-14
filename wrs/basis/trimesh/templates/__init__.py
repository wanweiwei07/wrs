from importlib.resources import read_text


def get_template(name):
    result = read_text('wrs.basis.trimesh.templates', name)
    if hasattr(result, 'decode'):
        return result.decode('utf-8')
    return result
