<div align="center">
<h1>3DGS-Viewer</h1>
    
![Teaser image](assets/demo.png)
</div>

## 😋Introduction
It is a simple 3DGS viewer based on viser. You can view a 3DGS model through a web browser without relying on a GUI. You can see the video in the ```assets/demo.mp4```.

## 🔨Installation
To install, you can use the following command:
```
pip install -r requirements.txt
pip install -e submodules/diff-gaussian-rasterization
pip install -e submodules/simple-knn
pip install -e submodules/fused-ssim
```
In contrast to the original 3DGS code, you need to add the following code to ```scene/cameras.py```:
```
class Simple_Camera(nn.Module):
    def __init__(self, colmap_id, R, T, FoVx, FoVy, h, w,
                 image_name, uid,
                 trans=np.array([0.0, 0.0, 0.0]), scale=1.0, data_device = "cuda", qvec=None
                 ):
        super(Simple_Camera, self).__init__()

        self.uid = uid
        self.colmap_id = colmap_id
        self.R = R
        self.T = T
        self.FoVx = FoVx
        self.FoVy = FoVy
        self.image_name = image_name
        self.qvec = qvec

        try:
            self.data_device = torch.device(data_device)
        except Exception as e:
            print(e)
            print(f"[Warning] Custom device {data_device} failed, fallback to default cuda device" )
            self.data_device = torch.device("cuda")

        self.image_width = w
        self.image_height = h


        self.zfar = 100.0
        self.znear = 0.01

        self.trans = trans
        self.scale = scale

        self.world_view_transform = torch.tensor(getWorld2View2(R, T, trans, scale)).transpose(0, 1).cuda()
        self.projection_matrix = getProjectionMatrix(znear=self.znear, zfar=self.zfar, fovX=self.FoVx, fovY=self.FoVy).transpose(0,1).cuda()
        self.full_proj_transform = (self.world_view_transform.unsqueeze(0).bmm(self.projection_matrix.unsqueeze(0))).squeeze(0)
        self.camera_center = self.world_view_transform.inverse()[3, :3]

    def HW_scale(self, h, w):
        return Simple_Camera(self.colmap_id, self.R, self.T, self.FoVx, self.FoVy, h, w, self.image_name, self.uid ,qvec=self.qvec)
```

## ⚡Run
To use the viewer, you can use the following command if you want to set colmap dir:
```
python webui.py --gs_source "<YOUR PATH>" --colmap_dir "<YOUR PATH>"
```
if you just have .ply file, you can use the following command:
```
python webui.py --gs_source "<YOUR PATH>"
```


## 😘Acknowledgements

This project is built upon [3D Gaussian Splatting](https://github.com/graphdeco-inria/gaussian-splatting) and [GaussianEditor](https://github.com/buaacyw/GaussianEditor). We want to thank the authors for their contributions.
