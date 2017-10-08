import os, sys

from PIL import Image
try:
    import accimage
except ImportError:
    accimage = None

from constant import ROOT_PATH


def _is_pil_image(img):
    if accimage is not None:
        return isinstance(img, (Image.Image, accimage.Image))
    else:
        return isinstance(img, Image.Image)

def resize(img, size, interpolation=Image.ANTIALIAS):
    """Resize the input PIL.Image to the given size.
    Args:
        img (PIL.Image): Image to be resized.
        size (sequence or int): Desired output size. If size is a sequence like
            (h, w), the output size will be matched to this. If size is an int,
            the smaller edge of the image will be matched to this number maintaing
            the aspect ratio. i.e, if height > width, then image will be rescaled to
            (size * height / width, size)
        interpolation (int, optional): Desired interpolation. Default is
            ``PIL.Image.ANTIALIAS``
    Returns:
        PIL.Image: Resized image.
    """
    if not _is_pil_image(img):
        raise TypeError('img should be PIL Image. Got {}'.format(type(img)))
    if not (isinstance(size, int) or (isinstance(size, collections.Iterable) and len(size) == 2)):
        raise TypeError('Got inappropriate size arg: {}'.format(size))

    if isinstance(size, int):
        w, h = img.size
        if (w <= h and w == size) or (h <= w and h == size):
            return img
        if w < h:
            ow = size
            oh = int(size * h / w)
            return img.resize((ow, oh), interpolation)
        else:
            oh = size
            ow = int(size * w / h)
            return img.resize((ow, oh), interpolation)
    else:
        return img.resize(size[::-1], interpolation)


def process(options, collection):
    rootpath = options.rootpath
    size = options.image_size
    id_path_file = os.path.join(rootpath, collection, 'id.imagepath.txt')
    data = [x.strip().split() for x in open(id_path_file).readlines() if x.strip()]
    num_images = len(data)

    for i,(imgid,impath) in enumerate(data):
        resfile = impath.replace('ImageData/', 'ImageData{}/'.format(size))
        output_dir = os.path.split(resfile)[0]
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        if os.path.exists(resfile):
            continue
        with open(impath, 'r+b') as f:
            with Image.open(f) as img:
                img = resize(img, size)        
                img.save(resfile, img.format)
        if i % 1000 == 0:
            print ("[%d/%d] Resized the images and saved into '%s'."
                   %(i, num_images, output_dir))        

def main(argv=None):
    if argv is None:
        argv = sys.argv[1:]

    from optparse import OptionParser
    parser = OptionParser(usage="""usage: %prog [options] collection""")
    parser.add_option("--overwrite", default=0, type="int", help="overwrite existing file (default: 0)")
    parser.add_option("--rootpath", default=ROOT_PATH, type="string", help="rootpath (default: %s)" % ROOT_PATH)
    parser.add_option("--image_size", default=256, type="int", help="size for image after processing (default: 256)")

    (options, args) = parser.parse_args(argv)
    if len(args) < 1:
        parser.print_help()
        return 1

    return process(options, args[0])



if __name__ == '__main__':
    sys.exit(main())

