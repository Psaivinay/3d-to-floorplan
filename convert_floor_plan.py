import cv2
import matplotlib.pyplot as plt
import os


def load_images(image_path):
    image = cv2.imread(image_path)
    return [image]


def extract_edges(images):
    edges = []
    for image in images:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        edge = cv2.Canny(thresh, 50, 150)
        edges.append(edge)
    return edges


def extract_contours(edges):
    contours = []
    for edge in edges:
        contours_found, _ = cv2.findContours(edge, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        contours.extend(contours_found)
    return contours


def main():
    # Load the image
    image_path = ('C:\\Users\\vinay punnamaraju pc\\PycharmProjects\\3d to 2d floor plans\\scripts\\images\\3d '
                  'image.jpg')
    images = load_images(image_path)

    # Extract edges from the image
    edges = extract_edges(images)

    # Extract contours from the edges
    contours = extract_contours(edges)

    # Create an output directory if it doesn't exist
    output_dir = 'output'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Visualize and save the contours
    for i, edge in enumerate(edges):
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(edge, (x, y), (x + w, y + h), (0, 255, 0), 2)
        plt.subplot(1, len(edges), i + 1)
        plt.imshow(edge)
        plt.title('Image {}'.format(i + 1))
        plt.xticks([]), plt.yticks([])
        plt.savefig(os.path.join(output_dir, 'image_{}.jpg'.format(i + 1)))
    plt.show()


if __name__ == "__main__":
    main()
