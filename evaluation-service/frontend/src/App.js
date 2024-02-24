// App.js
import React, { useState } from 'react';
import { Stage, Layer, Image, Line, Text } from 'react-konva';
import axios from 'axios';

const calculateScale = (containerSize, imageSize) => {
  return containerSize / imageSize;
};

const App = () => {
  const [imageName, setImageName] = useState(null);
  const [image, setImage] = useState(null);
  const [coordinates, setCoordinates] = useState([]);
  const [hoverPosition, setHoverPosition] = useState({ x: 0, y: 0 });

  const handleChange = (event) => {
    setImageName(event.target.value);
  };

  const handleMouseMove = (event) => {
    const { layerX, layerY } = event.evt;
    setHoverPosition({ x: layerX, y: layerY });
  };

  const handleImageLoad = () => {
    setCoordinates([]);
    const img = new window.Image();
    img.src = `http://127.0.0.1:5000/images/${imageName}`;
    img.onload = () => {
      const maxWidth = window.innerWidth * 0.8; // 80% of the screen width
      const scale = maxWidth / img.width;
      console.log('Image width:', img.width, 'Image height:', img.height, scale);
      setImage({ img, scale });
    };
  };

  const handleImageClick = (event) => {
    const { layerX, layerY } = event.evt;
    console.log('layerX:', layerX, 'layerY:', layerY);
    axios.post('http://127.0.0.1:5000/api/coordinates', { x: layerX / image.scale,  y: layerY / image.scale, imageName})
      .then(response => {
        console.log('Response:', response.data.coordinates);
        const scaledCoordinates = response.data.coordinates.map(coord => coord * image.scale);
        setCoordinates(scaledCoordinates);
      })
      .catch(error => {
        console.error('Error fetching coordinates:', error);
      });
  };

  return (
    <div>
      <input
        type="text"
        placeholder="Enter image name"
        value={imageName}
        onChange={handleChange}
      />
      <button onClick={handleImageLoad}>Load Image</button>
      <Stage onMouseMove={handleMouseMove} width={image ? image.img.width * image.scale : 800} height={image ? image.img.height * image.scale : 600}>
        <Layer>
          {image && (
            <Image
            image={image.img}
            onClick={handleImageClick}
            scaleX={image.scale}
            scaleY={image.scale}
          />
          )}
          <Polygon coordinates={coordinates} />
          {image && <HoverText x={hoverPosition.x} y={hoverPosition.y} scale={image.scale} />}
        </Layer>
      </Stage>
    </div>
  );
};

const HoverText = ({ x, y, scale }) => {
  return (
    <>
    <Text
      text={`Hover: (${x.toFixed(2)}, ${y.toFixed(2)})`}
      x={x - 150}
      y={y - 10}
      fontSize={12}
      backgroundColor="white"
      fill="green" // Set the fill color to red
    />
    <Text
      text={`Scaled: (${(x/scale).toFixed(2)}, ${(y/scale).toFixed(2)})`}
      x={x - 150}
      y={y - 30}
      fontSize={12}
      backgroundColor="white"
      fill="green" // Set the fill color to red
    />
    </>
  );
};

const Polygon = ({ coordinates }) => {
  if (coordinates.length < 3) return null;

  return (
      <Line
        points={coordinates}
        closed
        stroke="green"
        strokeWidth={2}
        fill="rgba(255, 0, 0, 0.5)" // Red color with 50% opacity
        opacity={0.5} // Additional opacity setting to ensure transparency
      />
  );
};

export default App;