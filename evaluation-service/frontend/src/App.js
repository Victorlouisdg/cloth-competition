// App.js
import React, { useState, useEffect } from 'react';
import { Stage, Layer, Image, Line, Text } from 'react-konva';
import axios from 'axios';
import Select from 'react-select';

const App = () => {
  const [datasetName, setDatasetName] = useState(null);
  const [image, setImage] = useState(null);
  const [coordinates, setCoordinates] = useState([]);
  const [hoverPosition, setHoverPosition] = useState({ x: 0, y: 0 });
  const [availableFiles, setAvailableFiles] = useState([]);


  useEffect(() => {
    fetchAvailableFiles();
  }, []);

  const fetchAvailableFiles = () => {
    axios.get('http://127.0.0.1:5000/api/datasets')
      .then(response => {
        setAvailableFiles(response.data.datasets.map(file => ({ value: file, label: file })));
      })
      .catch(error => {
        console.error('Error fetching available files:', error);
      });
  };

  const handleChange = (selectedOption) => {
    setDatasetName(selectedOption.value);
    handleImageLoad(selectedOption.value);
  };

  const handleMouseMove = (event) => {
    const { layerX, layerY } = event.evt;
    setHoverPosition({ x: layerX, y: layerY });
  };

  const handleImageLoad = (imageToLoad) => {
    setCoordinates([]);
    const img = new window.Image();
    img.src = `http://127.0.0.1:5000/datasets/${imageToLoad}`;
    img.onload = () => {
      const maxWidth = window.innerWidth * 0.7;
      const scale = maxWidth / img.width;
      console.log('Image width:', img.width, 'Image height:', img.height, scale);
      setImage({ img, scale });
    };
  };

  const handleImageClick = (event) => {
    const { layerX, layerY } = event.evt;
    console.log('layerX:', layerX, 'layerY:', layerY);
    axios.post('http://127.0.0.1:5000/api/coordinates', { x: layerX / image.scale,  y: layerY / image.scale, datasetName})
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
    <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center' }}>
      <div style={{ marginBottom: '20px' }}>
        <Select
          value={availableFiles.find(file => file.value === datasetName)}
          onChange={handleChange}
          options={availableFiles}
          placeholder="Select a data folder"
          styles={{
            container: (provided, state) => ({
              ...provided,
              width: '200px',
            }),
          }}
        />
      </div>
      {!!image && <div style={{ marginTop: '20px', border: '1px solid #ccc', borderRadius: '5px', padding: '10px' }}>
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
      </div>}
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