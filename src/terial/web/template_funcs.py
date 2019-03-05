from terial import config


def nginx_url(path):
    path = str(path).replace('/local1/data/terial', 'terial-static')
    path = str(path).replace('/projects/grail/kparnb/data/terial',
                             'terial-static')
    return config.WEB_ROOT + path


def _static_url(resource):
    resource = resource.lstrip('/')
    return f'{config.WEB_ROOT}terial-static/{resource}'


def get_pages(cur_page, n_total_pages, n_display=10):
    offsets = list(range(-n_display//2, n_display//2 + 1))
    candidates = [cur_page + offset for offset in offsets]
    if candidates[0] < 0:
        candidates = [c - candidates[0] for c in candidates]
    if candidates[-1] >= n_total_pages:
        candidates = [c - (candidates[-1] - n_total_pages) for c in candidates]

    candidates = [c for c in candidates if 0 <= c < n_total_pages]
    return candidates


def setup(jinja_env):
    jinja_env.globals['nginx_url'] = nginx_url
    jinja_env.globals['static'] = _static_url
    jinja_env.globals['get_pages'] = get_pages
