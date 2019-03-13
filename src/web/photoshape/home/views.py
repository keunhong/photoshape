import matlab.engine
from tempfile import NamedTemporaryFile
from django.http import HttpResponse
from django.shortcuts import render
from .forms import UploadedImageForm
from .forms import UploadedImageForm2
from .models import Material, UploadedImage
import os
import sys
from visdom import Visdom
import numpy as np
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../../../terial/classifier/inference')
# sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../../../../data/json')
import infer_one_web 
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../../../terial/')
from terial import config, controllers
from terial.flow import resize_flow, visualize_flow, apply_flow
from terial.models import ExemplarShapePair
from terial.database import session_scope
from terial.visutils import visualize_segment_map
import match_models
from pathlib import Path
from PIL import Image
import PIL.ImageOps 
import json
from django.views.decorators.csrf import csrf_exempt
import json
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../../../../thirdparty/rendkit/meshkit/')
import wavefront
import json
from django.template.loader import render_to_string
from django.core.files.storage import FileSystemStorage
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../../../../thirdparty/toolbox/')
from toolbox.images import mask_bbox, crop_tight_fg, visualize_map
from skimage.io import imsave, imread
from skimage.morphology import disk, binary_closing
from skimage import transform
from skimage.color import rgb2gray
import argparse
import os
import logging
import warnings
from typing import List
from tqdm import tqdm
from kitnn.colors import tensor_to_image, rgb_to_lab, image_to_tensor
from kitnn.utils import softmax2d
from pydensecrf import densecrf
#from terial.pairs.generate_warped_renderings import apply_segment_crf
# Create your views here.
vis = Visdom(env='compute-flows')
script_dir = os.path.dirname(os.path.realpath(__file__))
siftflow_path = Path(script_dir, '../../../../thirdparty/siftflow').resolve()

QUAL_COLORS = [
    (230, 25, 75),
    (60, 180, 75),
    (255, 225, 25),
    (0, 130, 200),
    (245, 130, 48),
    (145, 30, 180),
    (70, 240, 240),
    (240, 50, 230),
    (210, 245, 60),
    (250, 190, 190),
    (0, 128, 128),
    (230, 190, 255),
    (170, 110, 40),
    (255, 250, 200),
    (128, 0, 0),
    (170, 255, 195),
    (128, 128, 0),
    (255, 215, 180),
    (0, 0, 128),
    (128, 128, 128),
    (166, 206, 227),
    (31, 120, 180),
    (178, 223, 138),
    (51, 160, 44),
    (251, 154, 153),
    (227, 26, 28),
    (253, 191, 111),
    (255, 127, 0),
    (202, 178, 214),
    (106, 61, 154),
    (255, 255, 153),
    (177, 89, 40),
    (141, 211, 199),
    (255, 255, 179),
    (190, 186, 218),
    (251, 128, 114),
    (128, 177, 211),
    (253, 180, 98),
    (179, 222, 105),
    (252, 205, 229),
    (217, 217, 217),
    (188, 128, 189),
    (204, 235, 197),
    (255, 237, 111),
]


def apply_segment_crf(image, segment_map, theta_p=0.05, theta_L=5, theta_ab=5):
    image_lab = tensor_to_image(rgb_to_lab(image_to_tensor(image)))
    perc = np.percentile(np.unique(image[:, :, :3].min(axis=2)), 88)
    bg_mask = np.all(image > perc, axis=2)
    vis.image(bg_mask.astype(np.uint8) * 255, win='bg-mask')

    p_y, p_x = np.mgrid[0:image_lab.shape[0], 0:image_lab.shape[1]]

    feats = np.zeros((5, *image_lab.shape[:2]), dtype=np.float32)
    d = min(image_lab.shape[:2])
    feats[0] = p_x / (theta_p * d)
    feats[1] = p_y / (theta_p * d)
    # feats[2] = fg_mask / 50
    feats[2] = image_lab[:, :, 0] / theta_L
    feats[3] = image_lab[:, :, 1] / theta_ab
    feats[4] = image_lab[:, :, 2] / theta_ab
    # vals = [v for v in np.unique(segment_map) if v >= 0]
    vals = np.unique(segment_map)
    probs = np.zeros((*segment_map.shape, len(vals)))
    for i, val in enumerate(vals):
        probs[:, :, i] = segment_map == val
    probs[bg_mask, 0] = 3
    probs[~bg_mask & (segment_map == -1)] = 1 / (len(vals))
    probs = softmax2d(probs)

    # for c in range(probs.shape[2]):
    #     vis.image(probs[:, :, c], win=f'prob-{c}', opts={'title': f'prob-{c}'})

    crf = densecrf.DenseCRF2D(*probs.shape)
    unary = np.rollaxis(
        -np.log(probs), axis=-1).astype(dtype=np.float32, order='c')
    crf.setUnaryEnergy(np.reshape(unary, (probs.shape[-1], -1)))
    crf.addPairwiseEnergy(np.reshape(feats, (feats.shape[0], -1)),
                          compat=3)

    Q = crf.inference(20)
    Q = np.array(Q).reshape((-1, *probs.shape[:2]))
    probs = np.rollaxis(Q, 0, 3)

    cleaned_seg_ind_map = probs.argmax(axis=-1)
    cleaned_seg_map = np.full(cleaned_seg_ind_map.shape,
                              fill_value=-1, dtype=int)
    for ind in np.unique(cleaned_seg_ind_map):
        cleaned_seg_map[cleaned_seg_ind_map == ind] = vals[ind]
    return cleaned_seg_map


@csrf_exempt
def homepage(request):
    if request.method == 'POST':
        result = request.POST.get('result')
        form_id = request.POST.get('form_id')
        if result is not None and form_id is not None:
            try:
                img = UploadedImage.objects.get(pk=form_id)
                img.user_material = Material.objects.get(pk=result)
                img.save()
                return render(request, 'homepage.html', {'content': 'Your result has been submitted!'})
            except UploadedImage.DoesNotExist:
                return render(request, 'homepage.html', {'content': "Error: Object doesn't exist in dataset"})


    return render(request, 'homepage.html')

def homepage2(request):
        if request.method == 'POST':
                result = request.POST.get('result')
                form_id = request.POST.get('form_id')
                if result is not None and form_id is not None:
                        try:
                                img = UploadedImage.objects.get(pk=form_id)
                                img.user_material = Material.objects.get(pk=result)
                                img.save()
                                return render(request, 'homepage2.html', {'content': 'Your result has been submitted!'})
                        except UploadedImage.DoesNotExist:
                                return render(request, 'homepage2.html', {'content': "Error: Object doesn't exist in dataset"})


        return render(request, 'homepage2.html')

def bright_pixel_mask(image, percentile=80):
    image = rgb2gray(image)
    perc = np.percentile(np.unique(image), percentile)
    mask = image < perc
    return mask

@csrf_exempt
def display_models(request):
    print(request.POST)
    print(request.FILES)
    engine = matlab.engine.start_matlab()
    engine.addpath(str(siftflow_path))
    engine.addpath(str(siftflow_path / 'mexDenseSIFT'))
    engine.addpath(str(siftflow_path / 'mexDiscreteFlow'))
    if request.method == 'POST' and request.FILES['grayscale']:
        myfile = request.FILES['grayscale'] # absolute path 
        filename = myfile.name
        im = imread(myfile)[:, :, 0]
        im2 = imread('/homes/grail/xuyf/WindowsFolders/git/photoshape/src/web/photoshape'+request.POST['url'])
        im2 = im2[:, :, :3]
        #im2 = imread('original/chair1_1a9E6Qd.png')
        fg_mask = im > 0
        fg_bbox = mask_bbox(fg_mask)
        final_im = crop_tight_fg(im, config.SHAPE_REND_SHAPE ,bbox=fg_bbox, fill=0, order=0)
        res = filename.split(".");
        if len(res) > 0 and res[-1] != 'png':
            filename = ''.join(res[:-1])
            filename += '.png'
        imsave('/homes/grail/xuyf/WindowsFolders/git/photoshape/src/web/photoshape/images/grayscale/'+filename, final_im)
        with NamedTemporaryFile(suffix='.png') as exemplar_f, \
        NamedTemporaryFile(suffix='.png') as shape_f:
            base_pattern = np.dstack((np.zeros(config.SHAPE_REND_SHAPE), *np.meshgrid(np.linspace(0, 1, config.SHAPE_REND_SHAPE[0]),np.linspace(0, 1, config.SHAPE_REND_SHAPE[1]))))

            exemplar_sil = bright_pixel_mask(im2, percentile=80)
            exemplar_sil = binary_closing(exemplar_sil, selem=disk(3))
            exemplar_sil = transform.resize(exemplar_sil, (500, 500),anti_aliasing=True, mode='reflect')
            #shape_sil = pair.load_data(config.SHAPE_REND_SEGMENT_MAP_NAME) - 1
            shape_sil = (final_im > 0)
            shape_sil = binary_closing(shape_sil, selem=disk(3))

            exemplar_sil_im = exemplar_sil[:, :, None].repeat(repeats=3, axis=2).astype(float)
            shape_sil_im = shape_sil[:, :, None].repeat(repeats=3, axis=2).astype(float)

            exemplar_sil_im[exemplar_sil == 0] = base_pattern[exemplar_sil == 0]
            shape_sil_im[shape_sil == 0] = base_pattern[shape_sil == 0]

            vis.image(exemplar_sil_im.transpose((2, 0, 1)), win='exemplar-sil')
            vis.image(shape_sil_im.transpose((2, 0, 1)), win='shape-sil')

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                imsave(exemplar_f.name, exemplar_sil_im)
                imsave(shape_f.name, shape_sil_im)

            vx, vy = engine.siftflow(str(exemplar_f.name), str(shape_f.name),nargout=2)
            vx, vy = resize_flow(np.array(vx), np.array(vy),shape=config.SHAPE_REND_SHAPE)
        
        flow_vis = visualize_flow(vx, vy)

        vis.image(flow_vis.transpose((2, 0, 1)),
            win='sil-flow',
            opts={'title': 'sil-flow'})
        seg_vis = visualize_segment_map(final_im.astype(int) - 1)
        seg_vis_warped = apply_flow(seg_vis, vx, vy)
        
        shape_seg_map = final_im.astype(int) - 1
        warped_seg_map = apply_flow(shape_seg_map, vx, vy)
        
        vis.image(seg_vis.transpose((2, 0, 1)), win='seg-vis')
        vis.image(seg_vis_warped.transpose((2, 0, 1)), win='seg-vis-warped')
        #vis.image(warped_seg_map.transpose((2, 0, 1)), win='warped_seg_map')
        vis.image(((im2.astype(np.float32)/255 + seg_vis_warped)/2).transpose((2, 0, 1)),win='sil-flow-applied',
opts={'title': 'sil-flow-applied'})
        image = transform.resize(im2,
                             config.SHAPE_REND_SHAPE, anti_aliasing=True,
                             mode='reflect')
        crf_seg_map = apply_segment_crf(image, warped_seg_map)
        crf_seg_vis = visualize_segment_map(crf_seg_map)

        vis.image(crf_seg_vis.transpose((2, 0, 1)), win='crf_seg_vis')

        return HttpResponse("success")
    return HttpResponse("fail")

@csrf_exempt
def display_results2(request):
    if (request.method == 'POST'):
        print(request.POST)
        print(request.FILES)
        form = UploadedImageForm2(request.POST, request.FILES)
        if form.is_valid():
            form.save()   
            original = form.instance.original.name
            mask = form.instance.mask.name
            context={'url':form.instance.original.url}
            materials = infer_results(original, mask)
            # create or get Material instance
            for m_id, name in materials:
                m, _ = Material.objects.get_or_create(mid=m_id)
                form.instance.computed_materials.add(m)
            context['materials'] = materials
            context['form'] = form.instance.pk
            return HttpResponse(json.dumps(context))
        else:
            print(form.errors)
            print(form.non_field_errors())
            return HttpResponse(form.errors)

@csrf_exempt
def display_results(request):
    print(request.POST)
    print(request.FILES)

    if request.is_ajax():
        form = UploadedImageForm(request.POST, request.FILES)
        if form.is_valid():
            form.save()   
            original = form.instance.original.name
            s_id, phi, theta = find_match_models(original)

            #context['form'] = form.instance.pk

            mesh = wavefront.read_obj_file('/data/photoshape/blobs/shapes/'+ s_id +'/models/uvmapped_v2.obj')
            g_materials = []
            c_materials = []
            for idx, material in enumerate(mesh.materials):
                g_materials.append((material, idx+1))
                r, g, b = QUAL_COLORS[idx]
                c_materials.append((material, 'rgb('+str(r)+','+str(g)+','+str(b)+')'))
        
            html = render_to_string('models.html', {'models': [(s_id, phi, theta, mesh.bounding_size())], 'g_materials': g_materials, 'c_materials': c_materials, 'url':form.instance.original.url, 'name':original})
    
        else:
            print(form.errors)
            print(form.non_field_errors())
            html = render_to_string('models.html')

        return HttpResponse(html)

    return render(request, 'models.html')

def infer_results(original, mask):
    current_path = os.path.dirname(os.path.abspath(__file__))
    image_path = Path(current_path + '/../images/'+ original)
    mask_path = Path(current_path + '/../images/'+ mask)
    checkpoint_path= Path(current_path + '/../../../../data/classifier/model/model_best.pth.tar')

    res = mask.split(".");
    if len(res) > 0 and res[-1] != 'png':
        new_name = ''.join(res[:-1])
        new_name += '.png'
        new_mask_path = Path(current_path + '/../images/'+ new_name)
        os.rename(mask_path, new_mask_path)
        mask_path = new_mask_path

    image = Image.open(mask_path)

    if image.mode == 'RGBA':
        r,g,b,a = image.split()
        rgb_image = Image.merge('RGB', (r,g,b))

        inverted_image = PIL.ImageOps.invert(rgb_image)
        inverted_image = PIL.ImageOps.invert(inverted_image)

        r2,g2,b2 = inverted_image.split()

        final_transparent_image = Image.merge('RGBA', (r2,g2,b2,a))

        final_transparent_image.save(mask_path)

    else:
        inverted_image = PIL.ImageOps.invert(image)
        inverted_image.save(mask_path)

    img = Image.open(Path(current_path + '/../images/mask/black.png'))
    layer = Image.open(mask_path) # this file is the transparent one
    img.paste(layer, (0,0), mask=layer) 
    # the transparancy layer will be used as the mask
    img.save(mask_path)

    result = infer_one_web.start_infer(image_path, mask_path, checkpoint_path)

    with open(current_path + '/../../../../data/json/materials.json') as f:
        data = json.load(f)

    materials = []
    for material in result['material'][:5]:
        m_id = material['id']
        # m_substance = material['pred_substance']
        # m_url = '/images/materials/' + str(m_id) + '/images/previews/bmps.png'
        m_name = data[str(m_id)]['name']
        materials.append((m_id, m_name))
    return materials

def find_match_models(original):
    models = []
    current_path = os.path.dirname(os.path.abspath(__file__))
    image_path = current_path + '/../images/'+ original
    pairs = match_models.compute_pair(image_path)
    for pair in pairs[:1]:
        return str(pair.shape_id), pair.elevation, pair.azimuth
        #models.append((str(pair.shape_id), pair.elevation, pair.azimuth))
    return 


