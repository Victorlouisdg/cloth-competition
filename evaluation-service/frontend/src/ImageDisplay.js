import React, { useState, useRef } from "react";
import { Stage, Layer, Image, Rect, Circle } from "react-konva";

var colorReplaceFilter = function (imageData) {
    var nPixels = imageData.data.length;
    for (var i = 0; i < nPixels; i += 4) {
      const isWhite =
        imageData.data[i] === 255 &&
        imageData.data[i + 1] === 255 &&
        imageData.data[i + 2] === 255;
      if (isWhite) {
        imageData.data[i] = 0; // Red
        imageData.data[i + 1] = 255; // Green
        imageData.data[i + 2] = 0; // Blue
      }
    }
  };

const ImageDisplay = ({
    image,
    maskImage,
    depthImage,
    rect,
    setRect,
    positiveKeypoints,
    setPositiveKeypoints,
    negativeKeypoints,
    setNegativeKeypoints,
    annotationMode,
    showDepth,
    showMask,
    // handleAnnotate
}) => {
    const maskImageRef = React.useRef();
    const scale = image?.scale;
    const [drawing, setDrawing] = useState(false);


    // if image is None, return a placeholder div
    if (!image) {
        return <div>No image to display</div>;
    }

    const handleMouseDown = (e) => {
        if (annotationMode === "positive") {
            setPositiveKeypoints((points) => [
                ...points,
                e.target.getStage().getPointerPosition(),
            ]);
            return;
        }

        if (annotationMode === "negative") {
            setNegativeKeypoints((points) => [
                ...points,
                e.target.getStage().getPointerPosition(),
            ]);
            return;
        }

        setDrawing(true);
        const pos = e.target.getStage().getPointerPosition();
        setRect({ x: pos.x, y: pos.y, width: 10, height: 10 });
    };

    const handleMouseMove = (e) => {
        if (!drawing) return;
        const pos = e.target.getStage().getPointerPosition();
        setRect((rect) => ({
            ...rect,
            width: pos.x - rect.x,
            height: pos.y - rect.y,
        }));
    };

    const handleMouseUp = () => {
        setDrawing(false);
    };

    return (
        <Stage
            onMouseDown={handleMouseDown}
            onMouseMove={handleMouseMove}
            onMouseUp={handleMouseUp}
            width={image ? image.img.width * scale : 800}
            height={image ? image.img.height * scale : 600}
        >
            <Layer>
                {image && (
                    <Image
                        image={image.img}
                        scaleX={scale}
                        scaleY={scale}
                    />
                )}
                {/* ... (Similar logic for depthImage) */}
                {depthImage && showDepth && (
                    <Image
                        image={depthImage.img}
                        // onClick={handleImageClick}
                        scaleX={depthImage.scale}
                        scaleY={depthImage.scale}
                    />
                )}

                {maskImage && showMask && (
                    <Image
                        image={maskImage.img}
                        opacity={0.3}
                        scaleX={maskImage.scale}
                        scaleY={maskImage.scale}
                        filters={[colorReplaceFilter]}
                        ref={maskImageRef}
                    />
                )}

                {!!rect && <Rect {...rect} stroke="#007BFF" strokeWidth={4} />}
                {positiveKeypoints.map((pos, i) => (
                    <Circle key={i} x={pos.x} y={pos.y} radius={5} fill="green" />
                ))}
                {negativeKeypoints.map((pos, i) => (
                    <Circle key={i} x={pos.x} y={pos.y} radius={5} fill="red" />
                ))}
            </Layer>
        </Stage>
    );
};

export default ImageDisplay;