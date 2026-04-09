from manim import *
import pandas as pd
import numpy as np
import os

class PCASpectrogram(ThreeDScene):
    def construct(self):
        # 1. Load PCA Data
        # Assuming pca.csv is in the same directory as this script
        try:
            pca_df = pd.read_csv("output/pca/pca.csv")
        except FileNotFoundError:
            # Fallback dummy data for testing if pca.csv is missing
            pca_df = pd.DataFrame({
                "PC1": [7.23, -8.33, 1.09],
                "PC2": [4.09, 2.66, -6.76],
                "PC3": [0.0, 0.0, 0.0],
                "Filename": ["September.mp3", "505.mp3", "DoI.mp3"]
            })

        # 2. Setup 3D Axes
        axes = ThreeDAxes(
            x_range=[-10, 10, 2],
            y_range=[-10, 10, 2],
            z_range=[-2, 2, 1],
            x_length=8,
            y_length=8,
            z_length=4
        )
        
        # Add labels to the axes
        axes_labels = axes.get_axis_labels(
            Text("PC1").scale(0.5), 
            Text("PC2").scale(0.5), 
            Text("PC3").scale(0.5)
        )

        self.set_camera_orientation(phi=75 * DEGREES, theta=30 * DEGREES)
        self.play(Create(axes), FadeIn(axes_labels))
        self.wait(1)

        # 3. Plot PCA Points and Labels
        dots = VGroup()
        labels = VGroup()
        
        for index, row in pca_df.iterrows():
            # Get coordinates
            x, y, z = row['PC1'], row['PC2'], row['PC3']
            
            # Create 3D dot
            dot = Dot3D(point=axes.c2p(x, y, z), color=BLUE, radius=0.15)
            dots.add(dot)
            
            # Create label
            label = Text(row['Filename'], font_size=16).next_to(dot, UP)
            # Make label always face the camera
            self.add_fixed_orientation_mobjects(label)
            labels.add(label)

        self.play(Create(dots))
        self.play(FadeIn(labels))
        self.wait(2)
# 4. Animate through each song and show its spectrogram
        for index, row in pca_df.iterrows():
            filename = row['Filename']
            x, y, z = row['PC1'], row['PC2'], row['PC3']
            
            # Highlight current dot
            current_dot = dots[index]
            self.play(current_dot.animate.set_color(YELLOW).scale(1.5), run_time=0.5)
            
            # Move camera to focus on this point
            self.move_camera(
                phi=60 * DEGREES, 
                theta=45 * DEGREES, 
                focal_point=axes.c2p(x, y, z),
                run_time=1.5
            )

            # Generate Spectrogram Graphic
            spectrogram_group = self.create_spectrogram_plot(filename)
            
            # Lock to camera frame FIRST
            self.add_fixed_in_frame_mobjects(spectrogram_group)
            
            # Fade it in safely by animating opacity
            spectrogram_group.set_opacity(0)
            self.play(spectrogram_group.animate.set_opacity(1), run_time=1)
            self.wait(3)
            
            # Cleanup and reset
            self.play(spectrogram_group.animate.set_opacity(0), run_time=0.5)
            self.remove(spectrogram_group)
            self.play(current_dot.animate.set_color(BLUE).scale(1/1.5), run_time=0.5)
        # 4. Animate through each song and show its spectrogram
        for index, row in pca_df.iterrows():
            filename = row['Filename']
            x, y, z = row['PC1'], row['PC2'], row['PC3']
            
            # Highlight current dot
            current_dot = dots[index]
            self.play(current_dot.animate.set_color(YELLOW).scale(1.5), run_time=0.5)
            
            # Move camera to focus on this point
            self.move_camera(
                phi=60 * DEGREES, 
                theta=45 * DEGREES, 
                focal_point=axes.c2p(x, y, z),
                run_time=1.5
            )

            # Generate Spectrogram Graphic
            spectrogram_group = self.create_spectrogram_plot(filename)
            
            # Lock to camera frame FIRST
            self.add_fixed_in_frame_mobjects(spectrogram_group)
            
            # Fade it in safely by animating opacity
            spectrogram_group.set_opacity(0)
            self.play(spectrogram_group.animate.set_opacity(1), run_time=1)
            self.wait(3)
            
            # Cleanup and reset
            self.play(spectrogram_group.animate.set_opacity(0), run_time=0.5)
            self.remove(spectrogram_group)
            self.play(current_dot.animate.set_color(BLUE).scale(1/1.5), run_time=0.5)

        # Reset camera to global view
        self.move_camera(phi=75 * DEGREES, theta=30 * DEGREES, focal_point=ORIGIN, run_time=2)
        self.wait(2)

    def create_spectrogram_plot(self, filename):
        """
        Loads a pre-generated matplotlib spectrogram image (.jpg) 
        and displays it as a 2D overlay.
        """
        image_path = f"output/img/{filename}.png" 
        print(f"Path {image_path}")
        
        # Background box to frame the image nicely
        bg_box = Rectangle(width=7, height=5, color=BLACK, fill_opacity=0.85).to_corner(UL)
        title = Text(f"Spectrogram: {filename}", font_size=20, color=YELLOW).next_to(bg_box.get_top(), DOWN, buff=0.2)
        
        # FIX: Use Group instead of VGroup so it can hold raster images!
        plot_group = Group(bg_box, title)

        if os.path.exists(image_path):
            # Load the pre-rendered image
            spectrogram_img = ImageMobject(image_path)
            
            # Scale it to fit inside our background box and position it below the title
            spectrogram_img.height = 3.8  
            spectrogram_img.next_to(title, DOWN, buff=0.2)
            
            plot_group.add(spectrogram_img)
        else:
            # If the image doesn't exist, display a red warning
            missing_text = Text(f"Image {image_path} not found", color=RED, font_size=16).move_to(bg_box.get_center())
            plot_group.add(missing_text)

        return plot_group
