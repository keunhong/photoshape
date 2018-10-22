import terial.web.views.dataset_views
import terial.web.views.figure_views
from terial.web.views import (envmap_views, exemplar_views, material_views,
                              pair_views, shape_views, classifier_views,
                              evaluation_views)
from terial.web.views import index


def setup(app, prefix):
    app.router.add_route('GET', prefix + '/', index, name='index')

    #
    # Exemplars.
    #
    app.router.add_route('GET', prefix + '/exemplars/',
                         exemplar_views.list_exemplars,
                         name='exemplar_list')
    app.router.add_route('PUT', prefix + '/exemplars/{exemplar_id:\d*}/exclude',
                         exemplar_views.set_exclude,
                         name='exemplar_set_exclude')

    #
    # Pairs.
    #
    app.router.add_route('GET', prefix + '/pairs/',
                         pair_views.list_pairs,
                         name='pair_list')
    app.router.add_route('GET', prefix + '/pairs-kostas/',
                         pair_views.list_pairs_kostas,
                         name='pair_list_kostas')
    app.router.add_route('GET', prefix + '/pairs/{pair_id:\d*}',
                         pair_views.show_pair,
                         name='show_pair')
    app.router.add_route('GET',
                         prefix + '/pairs/{pair_id:\d*}.exemplar.uncropped.jpg',
                         pair_views.get_uncropped_exemplar,
                         name='pair_uncropped_exemplar')

    #
    # Shapes.
    #
    app.router.add_route('GET', prefix + '/shapes/',
                         shape_views.list_shapes,
                         name='shape_list')
    app.router.add_route('PUT', prefix + '/shapes/{shape_id:\d*}/exclude',
                         shape_views.set_exclude,
                         name='shape_set_exclude')

    #
    # Materials.
    #
    app.router.add_route('GET', prefix + '/materials',
                         material_views.list_materials,
                         name='material_list')
    app.router.add_route('GET', prefix + '/materials/tree',
                         material_views.material_tree,
                         name='material_tree')
    app.router.add_route('GET', prefix + '/materials/tree.json',
                         material_views.material_tree_json,
                         name='material_tree_json')
    app.router.add_route('PUT',
                         prefix + '/materials/{material_id:\d*}/default_scale',
                         material_views.set_default_scale,
                         name='material_set_default_scale')
    app.router.add_route('PUT',
                         prefix + '/materials/{material_id:\d*}/enabled',
                         material_views.set_enabled,
                         name='material_set_enabled')

    #
    # Envmaps.
    #
    app.router.add_route('GET', prefix + '/envmaps',
                         envmap_views.list_envmaps,
                         name='envmap_list')


    #
    # Classifier.
    #
    app.router.add_route(
        'GET', prefix + '/classifier/datasets/',
        terial.web.views.dataset_views.list_datasets,
        name='classifier_list_datasets')
    app.router.add_route(
        'GET', prefix + '/classifier/datasets/{dataset_name}/',
        terial.web.views.dataset_views.search_dataset_renderings,
        name='classifier_search_dataset')

    app.router.add_route(
        'GET', prefix + '/classifier/models/',
        classifier_views.list_models,
        name='classifier_list_models')

    app.router.add_route(
        'GET', prefix + '/classifier/models-ablation/',
        classifier_views.list_models_ablation,
        name='classifier_list_models_ablation')

    app.router.add_route(
        'GET', prefix + '/classifier/snapshots/',
        terial.web.views.dataset_views.list_snapshots,
        name='classifier_list_snapshots')
    app.router.add_route(
        'GET',
        prefix + '/classifier/inference/detailed/{snapshot}/{model}/{epoch}',
        classifier_views.show_inference_results_detailed,
        name='classifier_inference_results_detailed')
    app.router.add_route(
        'GET',
        prefix + '/classifier/inference/simple/{snapshot}/{model}/{epoch}',
        classifier_views.show_inference_results_simple,
        name='classifier_inference_results_simple')

    app.router.add_route(
        'GET',
        prefix + '/classifier/download-blend',
        classifier_views.download_blend_file,
        name='classifier_download_blend_file')

    #
    # Evaluation.
    #
    app.router.add_route(
        'GET', prefix + '/evaluation',
        terial.web.views.evaluation_views.landing_page,
        name='evaluation_landing_page')
    app.router.add_route(
        'GET', prefix + '/evaluation/annotate',
        terial.web.views.evaluation_views.annotate,
        name='evaluation_annotate')
    app.router.add_route(
        'GET', prefix + '/evaluation/stats',
        terial.web.views.evaluation_views.show_stats,
        name='evaluation_stats')
    app.router.add_route(
        'GET', prefix + '/evaluation/shape',
        terial.web.views.evaluation_views.get_job,
        name='evaluation_get_job')
    app.router.add_route(
        'POST', prefix + '/evaluation/annotation',
        terial.web.views.evaluation_views.post_annotation,
        name='evaluation_post_annotation')


    #
    # Figures.
    #


    app.router.add_route(
        'GET',
        prefix + '/figures/classifier-three/{snapshot}/{model}/{epoch}',
        terial.web.views.figure_views.show_prcs,
        name='classifier_inference_results_figure')
    app.router.add_route(
        'GET',
        prefix + '/figures/classifier-main/{snapshot}/{model}/{epoch}',
        terial.web.views.figure_views.show_inference_results_figure_2,
        name='classifier_inference_results_figure_2')
    app.router.add_route(
        'GET',
        prefix + '/figures/prcs/{resultset_id:\d*}',
        terial.web.views.figure_views.show_prcs,
        name='classifier_show_prcs')
    app.router.add_route(
        'GET',
        prefix + '/figures/materials',
        terial.web.views.figure_views.show_all_materials,
        name='figures_show_all_materials')
    app.router.add_route(
        'GET',
        prefix + '/figures/renderings',
        terial.web.views.figure_views.show_renderings,
        name='figures_show_rendernings')
    app.router.add_route(
        'GET',
        prefix + '/figures/exemplars',
        terial.web.views.figure_views.show_exemplars,
        name='figures_show_exemplars')
    app.router.add_route(
        'GET',
        prefix + '/figures/shapes',
        terial.web.views.figure_views.show_shapes,
        name='figures_show_shapes')

