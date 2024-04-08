// App.js
import React, { useState, useEffect } from 'react';
import { Stage, Layer, Image, Line, Text } from 'react-konva';
import axios from 'axios';
import Select from 'react-select';
import Switch from "react-switch";
import ClipLoader from "react-spinners/ClipLoader";

const App = () => {
  const [scene, setScene] = useState(null);
  const [image, setImage] = useState(null);
  const [coordinates, setCoordinates] = useState([]);
  const [hoverPosition, setHoverPosition] = useState({ x: 0, y: 0 });
  const [availableFiles, setAvailableFiles] = useState([]);

  const [toggleState, setToggleState] = useState(false);

  const handleToggle = () => {
    setToggleState(!toggleState);
  };

  useEffect(() => {
    let intervalId;

    const startInterval = async () => {
      if (toggleState) {
        intervalId = setInterval(async () => {
          const availableFiles = await getAvailableFiles();
          setAvailableFiles(availableFiles);
          const sceneToSet = availableFiles.sort((a, b) => a.lastModifiedTime - b.lastModifiedTime).find(file => !file.maskExists);
          console.log('Scene to set:', sceneToSet, 'Current scene:', scene);
          if (sceneToSet && sceneToSet.sceneName !== scene.sceneName) {
            console.log('Setting new scene:', sceneToSet);
            setScene(sceneToSet);
            setImage(null);
            setCoordinates([]);
          }
        }, 3000);
      }
    }

    startInterval();
  
    
  
    return () => {
      if (intervalId) {
        clearInterval(intervalId); 
      }
    };
  }, [toggleState, scene])

  useEffect(() => {
    const fetchAvailableFiles = async () => {
      const initialAvailableFiles = await getAvailableFiles();
      setAvailableFiles(initialAvailableFiles);
    }

    fetchAvailableFiles();
  }, []);

  useEffect(() => {
    const fetchSceneImageAndMaybeMask = async () => {
      console.log('Scene changed:', scene);
      if (!scene) return;

      if (scene.maskExists) {
        const coordinates = await axios.get(`http://127.0.0.1:5000/api/coordinates/${scene.sceneName}`);
        console.log('Coordinates:', coordinates.data.coordinates);
        setCoordinates(coordinates.data.coordinates);
      }
      const img = new window.Image();
      img.src = `http://127.0.0.1:5000/datasets/${scene.sceneName}`;
      img.onload = () => {
        const maxWidth = window.innerWidth * 0.7;
        const scale = maxWidth / img.width;
        console.log('Image width:', img.width, 'Image height:', img.height, scale);
        setImage({ img, scale });
      };
      
      
    }

    fetchSceneImageAndMaybeMask();
  }, [scene]);

  const getAvailableFiles = async () => {
    try {
      const response = await axios.get('http://127.0.0.1:5000/api/datasets');
      return response.data.datasets
    } catch (error) {
      console.error('Error fetching available files:', error);
      return [];
    }
  };

  const handleChange = (selectedOption) => {
    setImage(null);
    setCoordinates([]);
    setScene(selectedOption.value);
  };

  const handleMouseMove = (event) => {
    const { layerX, layerY } = event.evt;
    setHoverPosition({ x: layerX, y: layerY });
  };


  const handleImageClick = (event) => {
    const { layerX, layerY } = event.evt;
    console.log('layerX:', layerX, 'layerY:', layerY);
    axios.post('http://127.0.0.1:5000/api/annotate', { x: layerX / image.scale,  y: layerY / image.scale, sceneName: scene.sceneName})
      .then(response => {
        setCoordinates(response.data.coordinates);
      })
      .catch(error => {
        console.error('Error fetching coordinates:', error);
      });
  };

  const selectOptions = availableFiles.map(file => ({ value: file, label: file.sceneName }));

  const scaledCoordinates = !!image ? coordinates.map(coord => coord * image.scale) : [];

  return (
    <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center' }}>
      <div style={{ display: 'flex', flexDirection: 'row', alignItems: 'center', justifyContent: 'space-between', width: '50%' }}>
      
      <Select
          value={selectOptions.find(option => option.value.sceneName === scene?.sceneName)}
          onChange={handleChange}
          options={selectOptions}
          placeholder="Select data"
          styles={{
            container: (provided, state) => ({
              ...provided,
              width: '200px',
            }),
          }}
        />
      <div style={{ display: 'flex', flexDirection: 'row', alignItems: 'center' }}>
      <div style={{ marginRight: '10px', marginLeft: '10px' }}>
        Auto check for unannotated data
      </div>
        <Switch 
          onChange={handleToggle} 
          checked={toggleState} 
        />
      </div>
      </div>
      {!!image ? (<div style={{ marginTop: '20px', border: '1px solid #ccc', borderRadius: '5px', padding: '10px' }}>
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
            <Polygon coordinates={scaledCoordinates} />
            {image && <HoverText x={hoverPosition.x} y={hoverPosition.y} scale={image.scale} />}
          </Layer>
        </Stage>
      </div>) :
      <div style={{ marginTop: '200px' }}>
       {!!scene && <ClipLoader loading={true} size={20} />}
       </div>
       }
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