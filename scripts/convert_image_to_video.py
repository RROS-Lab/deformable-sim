import argparse
import glob

from moviepy import ImageSequenceClip


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("image_folder", type=str, help="Folder containing images")
    parser.add_argument(
        "-o", "--output", type=str, default="output", help="Output video file"
    )
    parser.add_argument("--fps", type=int, default=60, help="Frames per second")
    args = parser.parse_args()

    image_files = sorted(glob.glob(args.image_folder + "/*.png"))
    if not image_files:
        print("No images found in the specified folder.")
        return

    clip = ImageSequenceClip(image_files, fps=args.fps)
    clip.write_videofile(
        filename=args.output + ".mp4",
        fps=args.fps,
        codec="png",
    )

    clip.write_gif(
        filename=args.output + ".gif",
        fps=args.fps,
    )


if __name__ == "__main__":
    main()
