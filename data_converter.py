import argparse
import os
from tqdm import tqdm
# from utils.video_io import VideoReader, VideoWriter

import imageio
import os
import datetime
# from utils import is_image_file, is_video_file

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in ['jpg', 'jpeg', 'png'])


def is_video_file(filename):
    return any(filename.endswith(extension) for extension in ['mp4', 'mkv', 'avi', 'gif'])


class VideoReader:
    def __init__(self, path):
        if is_video_file(path):
            self.reader = imageio.get_reader(path)
        elif os.path.isdir(path):
            self.frame_path_list = [os.path.join(path, filename) for filename in sorted(os.listdir(path)) if is_image_file(filename)]
        else:
            raise ValueError('The path is supposed to be a video file or a image file directory.')

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        if hasattr(self, 'reader'):
            self.reader.close()

    def __iter__(self):
        return self.iter_data()

    def __len__(self):
        return self.get_length()

    def iter_data(self):
        """Iterate over all images in the series"""
        if hasattr(self, 'reader'):
            for image in self.reader:
                yield image
        else:
            for frame_path in self.frame_path_list:
                yield imageio.imread(frame_path)

    def get_length(self):
        """Get the number of frames in the file"""
        if hasattr(self, 'reader'):
            length = self.reader.get_length()
            if length == float('inf'):
                length = 0
                for _ in self.reader:
                    length += 1
        else:
            length = len(self.frame_path_list)
        return length

    def get_data(self, index):
        """Read frame data from the file, using the index"""
        return self.reader.get_data(index) if hasattr(self, 'reader') else imageio.imread(self.frame_path_list[index])


class VideoWriter:
    def __init__(self, path, output_format='mp4', fps=10, time_tag=True):
        self.base_path = os.path.splitext(path)[0]
        self.output_format = output_format

        if time_tag:
            time_tag = datetime.datetime.now().strftime("%H-%M-%S-%f")[:12]
            self.base_path += f'_{time_tag}'

        if is_video_file(output_format):
            output_path = f'{self.base_path}.{output_format}'
            self.writer = imageio.get_writer(output_path, fps=fps)
        elif is_image_file(output_format):
            self.cur_index = 0
            os.makedirs(self.base_path, exist_ok=True)
        else:
            raise ValueError('The output format is supposed to be video or image.')

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        if hasattr(self, 'writer'):
            self.writer.close()

    def append_data(self, frame):
        if hasattr(self, 'writer'):
            self.writer.append_data(frame)
        else:
            filename = f'{self.cur_index:06d}.{self.output_format}'
            output_path = os.path.join(self.base_path, filename)
            imageio.imwrite(output_path, frame)
            self.cur_index += 1


def process_video(args):
    video_name = os.path.basename(args.input_video).split('.')[0]
    output_base_path = os.path.join(args.output_dir, video_name)
    print(args.input_video)
    with VideoReader(args.input_video) as reader, VideoWriter(output_base_path, args.output_format, args.output_fps, time_tag=False) as writer:
        for frame in tqdm(reader, desc=f'Process video [{video_name}]'):
            writer.append_data(frame[..., :3])


def process_directory(args):
    file_list = os.listdir(args.input_dir)
    for file in file_list:
        video_name = os.path.basename(file).split('.')[0]
        input_video_path = os.path.join(args.input_dir, file)
        output_base_path = os.path.join(args.output_dir, video_name)
        with VideoReader(input_video_path) as reader, VideoWriter(output_base_path, args.output_format, args.output_fps, time_tag=False) as writer:
            for frame in tqdm(reader, desc=f'Process video [{video_name}]'):
                writer.append_data(frame[..., :3])


if __name__ == "__main__":
    # Parse argument
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_video", default='', type=str, help='Path to input video')
    parser.add_argument("--input_dir", default='input_dir', type=str, help='Path to input directory')
    parser.add_argument("--output_dir", default='output_dir', type=str, help='Path to output directory')
    parser.add_argument("--output_format", default='jpg', choices=['jpg', 'png', 'mp4', 'avi', 'gif'], help='Output format')
    parser.add_argument("--output_fps", default=10, type=int, help='FPS of output video')
    args = parser.parse_args()

    # Check and create directories
    if args.input_video:
        if not os.path.exists(args.input_video):
            raise ValueError(f'The input video "{args.input_video}" does not exist.')

        process_video(args)
    else:
        if not os.path.exists(args.input_dir):
            raise ValueError(f'The directory "{args.input_dir}" does not exist.')

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    process_directory(args)
        
'''usage
python data_converter.py --input_dir=test_data/character/puppy --output_dir=test_data/character/puppy --output_format=jpg
'''