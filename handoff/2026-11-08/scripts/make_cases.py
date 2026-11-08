from pathlib import Path
import json
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
B=Path(__file__).resolve().parents[1]; rr=json.loads((B/'cases/image_selection.json').read_text())
fig,axes=plt.subplots(3,4,figsize=(10,10.8))
for row,family in enumerate(['resnet','v2','clip']):
 selected=[r for r in rr if r['model']==family+'-20261031']
 for col,ax in enumerate(axes[row]):
  ax.axis('off')
  if col>=len(selected):
   ax.text(.5,.5,'No false positives\nfor V2 seed 20261031\non this test',ha='center',va='center',fontsize=11,transform=ax.transAxes);continue
  r=selected[col]; im=Image.open(B/'cases/images'/(r['image_sha256']+'.jpg'))
  ax.imshow(im,interpolation='nearest')
  kind='AI -> real (FN)' if r['label']==1 else 'Real -> AI (FP)'
  ax.set_title(f"{family.upper()} | {kind}\np(AI)={r['p_ai']:.5g} | t={r['threshold']:.3f}",fontsize=9)
  ax.text(.5,-.07,Path(r['path']).name,ha='center',va='top',transform=ax.transAxes,fontsize=8)
fig.suptitle('CIFAKE errors from fixed seed 20261031',fontsize=16,y=.99)
fig.text(.5,.015,'Three lowest-score false negatives and highest-score false positive per family.\n32 x 32 source images enlarged with nearest-neighbor display; not a random or representative sample.',ha='center',fontsize=10)
fig.subplots_adjust(left=.035,right=.98,bottom=.10,top=.91,hspace=.43,wspace=.22);fig.savefig(B/'figures/06_error_cases.png',dpi=170);fig.savefig(B/'figures/06_error_cases.pdf');plt.close(fig)
