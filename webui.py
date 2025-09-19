import time
from typing import Any, Dict
import numpy as np
import torch
from scene.cameras import Simple_Camera

import viser
import viser.transforms as tf
from viser.theme import TitlebarButton, TitlebarConfig, TitlebarImage

import numpy as np
import torch
from omegaconf import OmegaConf
# from torchvision.transforms.functional import to_pil_image, to_tensor

from gaussian_renderer import render
from scene import GaussianModel
from scene.camera_scene import CamScene

from arguments import (
    PipelineParams,
    OptimizationParams,
)

from argparse import ArgumentParser

import math

import os
import random
import ui_utils

import datetime
from pathlib import Path

class WebUI:
    def __init__(self, cfg) -> None:
        self.gs_source = cfg.gs_source
        self.colmap_dir = cfg.colmap_dir
        self.port = 8084

        self.gaussian = GaussianModel(
            sh_degree=3,
        )
        # load
        self.gaussian.load_ply(self.gs_source)
        self.gaussian.max_radii2D = torch.zeros(
            (self.gaussian.get_xyz.shape[0]), device="cuda"
        )
        # front end related
        self.colmap_cameras = None
        self.render_cameras = None

        if self.colmap_dir is not None:
            scene = CamScene(self.colmap_dir, h=512, w=512)
            self.cameras_extent = scene.cameras_extent
            self.colmap_cameras = scene.cameras

        self.background_tensor = torch.tensor(
            [0, 0, 0], dtype=torch.float32, device="cuda"
        )

        self.parser = ArgumentParser(description="Training script parameters")
        self.pipe = PipelineParams(self.parser)

        self.viewer_need_update = False
        self.system_need_update = False
        self.inpaint_again = True
        self.scale_depth = True

        self.server = viser.ViserServer(port=self.port)
        self.add_theme()
        self.draw_flag = True
        with self.server.add_gui_folder("Render Setting"):
            self.resolution_slider = self.server.add_gui_slider(
                "Resolution", min=384, max=4096, step=2, initial_value=768
            )

            self.FoV_slider = self.server.add_gui_slider(
                "FoV Scaler", min=0.2, max=2, step=0.1, initial_value=1
            )

            self.fps = self.server.add_gui_text(
                "FPS", initial_value="-1", disabled=True
            )
            self.renderer_output = self.server.add_gui_dropdown(
                "Renderer Output",
                [
                    "comp_rgb",
                ],
            )
            self.save_button = self.server.add_gui_button("Save Gaussian")

            self.frame_show = self.server.add_gui_checkbox(
                "Show Frame", initial_value=False
            )

        @self.save_button.on_click
        def _(_):
            current_time = datetime.datetime.now()
            formatted_time = current_time.strftime("%Y-%m-%d-%H:%M")
            self.gaussian.save_ply(os.path.join("ui_result", "{}.ply".format(formatted_time)))

        with torch.no_grad():
            self.frames = []
            random.seed(0)
            frame_index = random.sample(
                range(0, len(self.colmap_cameras)),
                min(len(self.colmap_cameras), 20),
            )
            for i in frame_index:
                self.make_one_camera_pose_frame(i)

        @self.frame_show.on_update
        def _(_):
            for frame in self.frames:
                frame.visible = self.frame_show.value
            self.server.world_axes.visible = self.frame_show.value

    def make_one_camera_pose_frame(self, idx):
        cam = self.colmap_cameras[idx]
        # wxyz = tf.SO3.from_matrix(cam.R.T).wxyz
        # position = -cam.R.T @ cam.T

        T_world_camera = tf.SE3.from_rotation_and_translation(
            tf.SO3(cam.qvec), cam.T
        ).inverse()
        wxyz = T_world_camera.rotation().wxyz
        position = T_world_camera.translation()

        # breakpoint()
        frame = self.server.add_frame(
            f"/colmap/frame_{idx}",
            wxyz=wxyz,
            position=position,
            axes_length=0.2,
            axes_radius=0.01,
            visible=False,
        )
        self.frames.append(frame)

        @frame.on_click
        def _(event: viser.GuiEvent):
            client = event.client
            assert client is not None
            T_world_current = tf.SE3.from_rotation_and_translation(
                tf.SO3(client.camera.wxyz), client.camera.position
            )

            T_world_target = tf.SE3.from_rotation_and_translation(
                tf.SO3(frame.wxyz), frame.position
            ) @ tf.SE3.from_translation(np.array([0.0, 0.0, -0.5]))

            T_current_target = T_world_current.inverse() @ T_world_target

            for j in range(5):
                T_world_set = T_world_current @ tf.SE3.exp(
                    T_current_target.log() * j / 4.0
                )

                with client.atomic():
                    client.camera.wxyz = T_world_set.rotation().wxyz
                    client.camera.position = T_world_set.translation()

                time.sleep(1.0 / 15.0)
            client.camera.look_at = frame.position

        if not hasattr(self, "begin_call"):

            def begin_trans(client):
                assert client is not None
                T_world_current = tf.SE3.from_rotation_and_translation(
                    tf.SO3(client.camera.wxyz), client.camera.position
                )

                T_world_target = tf.SE3.from_rotation_and_translation(
                    tf.SO3(frame.wxyz), frame.position
                ) @ tf.SE3.from_translation(np.array([0.0, 0.0, -0.5]))

                T_current_target = T_world_current.inverse() @ T_world_target

                for j in range(5):
                    T_world_set = T_world_current @ tf.SE3.exp(
                        T_current_target.log() * j / 4.0
                    )

                    with client.atomic():
                        client.camera.wxyz = T_world_set.rotation().wxyz
                        client.camera.position = T_world_set.translation()
                client.camera.look_at = frame.position

            self.begin_call = begin_trans

    def configure_optimizers(self):
        opt = OptimizationParams(
            parser = ArgumentParser(description="Training script parameters"),
            max_steps= self.edit_train_steps.value,
            lr_scaler = self.gs_lr_scaler.value,
            lr_final_scaler = self.gs_lr_end_scaler.value,
            color_lr_scaler = self.color_lr_scaler.value,
            opacity_lr_scaler = self.opacity_lr_scaler.value,
            scaling_lr_scaler = self.scaling_lr_scaler.value,
            rotation_lr_scaler = self.rotation_lr_scaler.value,

        )
        opt = OmegaConf.create(vars(opt))
        # opt.update(self.training_args)
        self.gaussian.spatial_lr_scale = self.cameras_extent
        self.gaussian.training_setup(opt)

    def render(
        self,
        cam,
        local=False,
        train=False,
    ) -> Dict[str, Any]:
        self.gaussian.localize = local

        render_pkg = render(cam, self.gaussian, self.pipe, self.background_tensor)
        image, viewspace_point_tensor, _, radii = (
            render_pkg["render"],
            render_pkg["viewspace_points"],
            render_pkg["visibility_filter"],
            render_pkg["radii"],
        )
        if train:
            self.viewspace_point_tensor = viewspace_point_tensor
            self.radii = radii
            self.visibility_filter = self.radii > 0.0

        image = image.permute(1, 2, 0)[None]  # C H W to 1 H W C
        render_pkg["comp_rgb"] = image  # 1 H W C

        depth = render_pkg["depth"]
        depth = depth.permute(1, 2, 0)[None]
        render_pkg["depth"] = depth
        render_pkg["opacity"] = depth / (depth.max() + 1e-5)

        return {
            **render_pkg,
        }

    @property
    def camera(self):
        if len(list(self.server.get_clients().values())) == 0:
            return None
        if self.render_cameras is None and self.colmap_dir is not None:
            self.aspect = list(self.server.get_clients().values())[0].camera.aspect
            self.render_cameras = CamScene(
                self.colmap_dir, h=-1, w=-1, aspect=self.aspect
            ).cameras
            self.begin_call(list(self.server.get_clients().values())[0])
        viser_cam = list(self.server.get_clients().values())[0].camera
        # viser_cam.up_direction = tf.SO3(viser_cam.wxyz) @ np.array([0.0, -1.0, 0.0])
        # viser_cam.look_at = viser_cam.position
        R = tf.SO3(viser_cam.wxyz).as_matrix()
        T = -R.T @ viser_cam.position
        # T = viser_cam.position
        if self.render_cameras is None:
            fovy = viser_cam.fov * self.FoV_slider.value
        else:
            fovy = self.render_cameras[0].FoVy * self.FoV_slider.value

        fovx = 2 * math.atan(math.tan(fovy / 2) * self.aspect)
        # fovy = self.render_cameras[0].FoVy
        # fovx = self.render_cameras[0].FoVx
        # math.tan(self.render_cameras[0].FoVx / 2) / math.tan(self.render_cameras[0].FoVy / 2)
        # math.tan(fovx/2) / math.tan(fovy/2)

        # aspect = viser_cam.aspect
        width = int(self.resolution_slider.value)
        height = int(width / self.aspect)
        return Simple_Camera(0, R, T, fovx, fovy, height, width, "", 0)

    def set_system(self, system):
        self.system = system

    @torch.no_grad()
    def prepare_output_image(self, output):
        out_key = self.renderer_output.value
        out_img = output[out_key][0]  # H W C

        if out_img.dtype == torch.float32:
            out_img = out_img.clamp(0, 1)
            out_img = (out_img * 255).to(torch.uint8).cpu().to(torch.uint8)
            out_img = out_img.moveaxis(-1, 0)  # C H W

        self.renderer_output.options = list(output.keys())
        return out_img.cpu().moveaxis(0, -1).numpy().astype(np.uint8)

    def render_loop(self):
        while True:
            # if self.viewer_need_update:
            self.update_viewer()
            time.sleep(1e-2)

    @torch.no_grad()
    def update_viewer(self):
        gs_camera = self.camera
        if gs_camera is None:
            return
        output = self.render(gs_camera)

        out = self.prepare_output_image(output)
        self.server.set_background_image(out, format="jpeg")

    def add_theme(self):
        buttons = (
            TitlebarButton(
                text="Getting Started",
                icon=None,
                href="https://github.com/Devin100086/3DGS_Viewer/blob/master/docs/webui.md",
            ),
            TitlebarButton(
                text="Github",
                icon="GitHub",
                href="https://github.com/Devin100086/3DGS_Viewer",
            ),
            TitlebarButton(
                text="Devin",
                icon=None,
                href="https://github.com/Devin100086",
            ),
        )
        image = TitlebarImage(
            image_url_light="https://raw.githubusercontent.com/Devin100086/3DGS-Viewer/refs/heads/main/assert/logo.png",
            image_alt="3DGS Viewer Logo",
            href="https://github.com/Devin100086/3DGS_Viewer/",
        )
        titlebar_theme = TitlebarConfig(buttons=buttons, image=image)
        brand_color = self.server.add_gui_rgb("Brand color", (7, 0, 8), visible=False)

        self.server.configure_theme(
            titlebar_content=titlebar_theme,
            show_logo=True,
            brand_color=brand_color.value,
        )


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--gs_source", type=str, required=True)  # gs ply or obj file?
    parser.add_argument("--colmap_dir", type=str, required=True)  #

    args = parser.parse_args()
    webui = WebUI(args)
    webui.render_loop()