// App.js
import './App.css';
import React, { useState, useEffect } from 'react';
import { Stage, Layer, Image, Line, Text } from 'react-konva';
import Konva from 'konva';
import axios from 'axios';
import Select from 'react-select';
import Switch from "react-switch";
import ClipLoader from "react-spinners/ClipLoader";
import Modal from 'react-modal';

var ColorReplaceFilter = function (imageData) {
  console.log('Applying filter');
  var nPixels = imageData.data.length;
  for (var i = 0; i < nPixels; i += 4) {
    const isWhite = imageData.data[i] === 255 && imageData.data[i + 1] === 255 && imageData.data[i + 2] === 255;
    if (isWhite) {
      imageData.data[i] = 0;     // Red
      imageData.data[i + 1] = 255; // Green
      imageData.data[i + 2] = 0;   // Blue
    }
  }
};

const App = () => {
  const [isLoading, setIsLoading] = useState(false);
  const [showDepth, setShowDepth] = useState(false);
  const [showMask, setShowMask] = useState(false);
  const [scene, setScene] = useState(null);
  const [image, setImage] = useState(null);
  const [maskImage, setMaskImage] = useState(null);
  const [depthImage, setDepthImage] = useState(null);
  const [hoverPosition, setHoverPosition] = useState({ x: 0, y: 0 });
  const [availableFiles, setAvailableFiles] = useState([]);

  const [autoLoad, setAutoLoad] = useState(false);

  const maskImageRef = React.useRef();

  // when image is loaded we need to cache the shape
  React.useEffect(() => {
    if (maskImage) {
      // you many need to reapply cache on some props changes like shadow, stroke, etc.
      maskImageRef.current?.cache();
    }
  }, [maskImage, showMask]);

  const fetchSceneImageAndMaybeMask = async (currentScene) => {

    setIsLoading(true);

    try {
      const maxWidth = window.innerWidth * 0.8;

      if (currentScene.maskExists) {
        const maskImg = new window.Image();
        maskImg.src = `http://127.0.0.1:5000/scenes/${currentScene.sceneName}/mask?${Date.now()}`;
        maskImg.crossOrigin = 'Anonymous';
        maskImg.onload = () => {
          const scale = maxWidth / maskImg.width;
          setMaskImage({ img: maskImg, scale });
        };
      }

      const img = new window.Image();
      img.src = `http://127.0.0.1:5000/scenes/${currentScene.sceneName}/image`;
      img.crossOrigin = 'Anonymous';
      img.onload = () => {
        const scale = maxWidth / img.width;
        setImage({ img, scale });
      };

      const depthImg = new window.Image();
      depthImg.src = `http://127.0.0.1:5000/scenes/${currentScene.sceneName}/depth`;
      depthImg.crossOrigin = 'Anonymous';
      depthImg.onload = () => {
        const scale = maxWidth / depthImg.width;
        setDepthImage({ img: depthImg, scale });
      };
      
    } catch (error) {
      console.error('Error fetching scene image:', error);
    } finally {
      setIsLoading(false);
    }
  }

  const fetchAvailableFiles = async () => {
    setIsLoading(true);
    try {
      const initialAvailableFiles = await getAvailableFiles();
      setAvailableFiles(initialAvailableFiles);
    } catch (error) {
      console.error('Error fetching available files:', error);
    } finally {
      setIsLoading(false);
    }
    
  }


  const handleRefresh = async () => {
    fetchAvailableFiles();
    if (scene) {
      fetchSceneImageAndMaybeMask(scene);
    }
  }

  // useEffect(() => {
  //   let intervalId;

  //   const startInterval = async () => {
  //     if (autoLoad) {
  //       intervalId = setInterval(async () => {
  //         const availableFiles = await getAvailableFiles();
  //         setAvailableFiles(availableFiles);
  //         const sceneToSet = availableFiles.sort((a, b) => a.lastModifiedTime - b.lastModifiedTime).find(file => !file.maskExists);
  //         console.log('Scene to set:', sceneToSet, 'Current scene:', scene);
  //         if (sceneToSet && sceneToSet.sceneName !== scene.sceneName) {
  //           console.log('Setting new scene:', sceneToSet);
  //           setScene(sceneToSet);
  //           setImage(null);
  //           setMask(null);
  //           // setCoordinates([]);
  //         }
  //       }, 3000);
  //     }
  //   }

  //   startInterval();
  
  //   return () => {
  //     if (intervalId) {
  //       clearInterval(intervalId); 
  //     }
  //   };
  // }, [autoLoad, scene])

  useEffect(() => {
    fetchAvailableFiles();
  }, []);

  useEffect(() => {
    if (scene) {
      fetchSceneImageAndMaybeMask(scene);
    }
    
  }, [scene]);

  const getAvailableFiles = async () => {
    try {
      const response = await axios.get('http://127.0.0.1:5000/api/scenes');
      return response.data.scenes
    } catch (error) {
      console.error('Error fetching available files:', error);
      return [];
    }
  };

  const handleChange = (selectedOption) => {
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
        fetchSceneImageAndMaybeMask(scene);
      })
      .catch(error => {
        console.error('Error fetching coordinates:', error);
      })
  };

  const selectOptions = availableFiles.map(file => ({ value: file, label: file.sceneName }));

  return (
  <>
    <Modal
      isOpen={isLoading}
      contentLabel="Loading Modal"
      ariaHideApp={false}
      style={{
        overlay: {
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          backgroundColor: 'rgba(0, 0, 0, 0.5)'
        },
        content: {
          position: 'absolute',
          top: '50%',
          left: '50%',
          background: 'none',
          border: 'none',
          padding: '0'
        }
      }}
    >
        <ClipLoader color="#ffffff" loading={true} size={30} />
    </Modal>
    
    
      <div style={{ display: 'flex', flexDirection: 'row'}}>
      <img style={{height: 60, width: 100}} src={process.env.PUBLIC_URL + '/ICRA_2024.jpg'} alt="Logo" className="logo" />
      
      <div style={{ display: 'flex', flexDirection: 'row', alignItems: 'space-between' }}>
      <div style={{ display: 'flex', flexDirection: 'row', alignItems: 'center' }}>
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

        <button className="refresh-button" onClick={handleRefresh}>Refresh</button>
      </div>

      <div style={{ display: 'flex', flexDirection: 'row', alignItems: 'center' }}>
        <Switch 
            onChange={() => {
              setShowMask(!showMask)
            }} 
            checked={showMask} 
          />
        <div className="switch-label">
          Show Mask
        </div>
        <Switch 
            onChange={() => setShowDepth(!showDepth)} 
            checked={showDepth} 
          />
        <div className="switch-label">
          Show Depth
        </div>
        
        <Switch 
            onChange={() => setAutoLoad(!autoLoad)} 
            checked={autoLoad} 
          />
        <div className="switch-label">
          Auto check for new data
        </div>
      </div>
      </div>
      
      
      </div>
      <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center'}}>
      {!!image && (<div style={{ marginTop: '20px', border: '1px solid #ccc', borderRadius: '5px', padding: '10px' }}>
        <Stage onMouseMove={handleMouseMove} width={image ? image.img.width * image.scale : 800} height={image ? image.img.height * image.scale : 600}>
          <Layer>
            {image && !showDepth && (
              <Image
                image={image.img}
                onClick={handleImageClick}
                scaleX={image.scale}
                scaleY={image.scale}
              />
            )}
            {depthImage && showDepth && (
              <Image
                image={depthImage.img}
                onClick={handleImageClick}
                scaleX={depthImage.scale}
                scaleY={depthImage.scale}
              />
            )}
            {maskImage && showMask && (
              <Image
                onClick={handleImageClick}
                image={maskImage.img}
                opacity={0.3}
                scaleX={maskImage.scale}
                scaleY={maskImage.scale}
                filters={[ColorReplaceFilter]}
                ref={maskImageRef}
              />
            )}
            <HoverText x={hoverPosition.x} y={hoverPosition.y} scale={image.scale} />
          </Layer>
        </Stage>
      </div>)}
    </div>
    </>
  );
};

const HoverText = ({ x, y, scale }) => {
  return (
    <>
    <Text
      text={`Pixel: (${(x/scale).toFixed(2)}, ${(y/scale).toFixed(2)})`}
      x={x - 150}
      y={y - 30}
      fontSize={12}
      backgroundColor="white"
      fill="green" // Set the fill color to red
    />
    </>
  );
};


export default App;