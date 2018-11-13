from django.shortcuts import render
from django.http import HttpResponse
from .forms import UploadedImageForm
import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../../../terial/classifier/inference')
import infer_one_web 
from pathlib import Path
from PIL import Image
import PIL.ImageOps 
import json
# Create your views here.

def homepage(request):
	return render(request, 'homepage.html')

def display_results(request):
	if (request.method == 'POST'):
		form = UploadedImageForm(request.POST, request.FILES)
		if form.is_valid():
			form.save()   
			original = form.instance.original.name
			mask = form.instance.mask.name
			context={'url':form.instance.original.url}
			materials = infer_results(original, mask)
			context['materials'] = materials
			return HttpResponse(json.dumps(context))
		else:
			print(form.errors)
			print(form.non_field_errors())
			return HttpResponse(form.errors)

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

	    r2,g2,b2 = inverted_image.split()

	    final_transparent_image = Image.merge('RGBA', (r2,g2,b2,a))

	    final_transparent_image.save(mask_path)

	else:
	    inverted_image = PIL.ImageOps.invert(image)
	    inverted_image.save(mask_path)

	img = Image.open(Path(current_path + '/../images/mask/black.png'))
	layer = Image.open(mask_path) # this file is the transparent one
	print(mask_path)
	img.paste(layer, (0,0), mask=layer) 
	# the transparancy layer will be used as the mask
	img.save(mask_path)

	result = infer_one_web.start_infer(image_path, mask_path, checkpoint_path)
	materials = []
	for material in result['material'][:5]:
		m_id = material['id']
		# m_substance = material['pred_substance']
		# m_url = '/images/materials/' + str(m_id) + '/images/previews/bmps.png'
		materials.append(m_id)
	print(materials)
	return materials

