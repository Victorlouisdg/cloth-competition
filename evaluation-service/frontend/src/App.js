// App.js
import "./App.css";
import React, { useState, useEffect, useRef } from "react";
import axios from "axios";
import Select from "react-select";
import Switch from "react-switch";
import ClipLoader from "react-spinners/ClipLoader";
import Modal from "react-modal";
import _, { set } from "lodash";
import SampleSelection from "./SampleSelection";
import ImageDisplay from "./ImageDisplay";


const customStyles = {
  option: (provided, state) => ({
    ...provided,
    color: "black",
    backgroundColor: state.isSelected ? "lightGrey" : "white",
  }),
  control: (provided) => ({
    ...provided,
    backgroundColor: "white",
    color: "black",
  }),
};

const App = () => {
  const [isLoading, setIsLoading] = useState(true);
  const [annotationIsLoading, setAnnotationIsLoading] = useState(false);

  const [showDepth, setShowDepth] = useState(false);
  const [showMask, setShowMask] = useState(false);
  const [autoLoad, setAutoLoad] = useState(false);

  const [currentTeam, setCurrentTeam] = useState(null);
  const [currentSample, setCurrentSample] = useState(null);
  const [currentObservation, setCurrentObservation] = useState(null);

  const [image, setImage] = useState(null);
  const [maskImage, setMaskImage] = useState(null);
  const [depthImage, setDepthImage] = useState(null);

  const [availableTeams, setAvailableTeams] = useState([]);
  const [availableSamples, setAvailableSamples] = useState([]);

  const [rect, setRect] = useState(null);
  const [drawing, setDrawing] = useState(false);
  const [outlierThreshold, setOutlierThreshold] = useState(null);

  const [positiveKeypoints, setPositiveKeypoints] = useState([]);
  const [negativeKeypoints, setNegativeKeypoints] = useState([]);

  const [annotationMode, setAnnotationMode] = useState("positive");

  const maskImageRef = React.useRef();

  const scale = image?.scale;

  const setAnnotationsFromResult = (result, image) => {

    if (!result || !image?.scale) return;

    setRect({
      x: result.bbox.x1 * scale,
      y: result.bbox.y1 * scale,
      width: (result.bbox.x2 - result.bbox.x1) * scale,
      height: (result.bbox.y2 - result.bbox.y1) * scale,
    });

    setPositiveKeypoints(
      result.positiveKeypoints.map((pos) => ({
        x: pos[0] * scale,
        y: pos[1] * scale,
      }))
    );

    setNegativeKeypoints(
      result.negativeKeypoints.map((pos) => ({
        x: pos[0] * scale,
        y: pos[1] * scale,
      }))
    );

  }

  useEffect(() => {
    const result = currentSample?.result;
    setAnnotationsFromResult(result, image);
  }, [image, currentSample]);

  React.useEffect(() => {
    if (maskImage) {
      maskImageRef.current?.cache();
    }
  }, [maskImage, showMask]);

  const updateImageSources = (sceneName) => {
    const maxWidth = window.innerWidth * 0.8;

    try {
      const maskImg = new window.Image();
      maskImg.src = `http://127.0.0.1:5000/scenes/${sceneName}/mask?${Date.now()}`;
      maskImg.crossOrigin = "Anonymous";
      maskImg.onload = () => {
        const scale = maxWidth / maskImg.width;
        setMaskImage({ img: maskImg, scale });
      };
    } catch (error) {
      console.error("Error fetching mask image:", error);
    }

    const img = new window.Image();
    img.src = `http://127.0.0.1:5000/scenes/${sceneName}/image`;
    img.crossOrigin = "Anonymous";
    img.onload = () => {
      const scale = maxWidth / img.width;
      setImage({ img, scale });
    };

    const depthImg = new window.Image();
    depthImg.src = `http://127.0.0.1:5000/scenes/${sceneName}/depth`;
    depthImg.crossOrigin = "Anonymous";
    depthImg.onload = () => {
      const scale = maxWidth / depthImg.width;
      setDepthImage({ img: depthImg, scale });
    };
  };

  const updateAvailableData = async () => {
    try {
      const availableTeamsResponse = await getAvailableTeams();
      // log the response
      console.log(availableTeamsResponse);
      setAvailableTeams(availableTeamsResponse);
      const availableScenesResponse = await getAvailableScenes();
      setAvailableSamples(availableScenesResponse);
    } catch (error) {
      console.error("Error fetching available files:", error);
    } finally {
      if (isLoading) {
        setIsLoading(false);
      }
    }
  };

  const handleRefresh = async () => {
    updateAvailableData();
    setAnnotationsFromResult(currentSample?.result, image);
  };

  const handleClearManualAnnotations = async () => {
    setPositiveKeypoints([]);
    setNegativeKeypoints([]);
    setRect(null);
  };

  const handleOutlierThresholdChange = (event) => {
    const outlierThreshold = event.target.value;
    setOutlierThreshold(outlierThreshold);
  };

  useEffect(() => {
    let intervalId;

    const startInterval = async () => {
      intervalId = setInterval(async () => {
        updateAvailableData();
      }, 3000);
    };

    startInterval();

    return () => {
      if (intervalId) {
        clearInterval(intervalId);
      }
    };
  }, []);

  // useEffect(() => {
  //   const sortedScenes = availableSamples.sort(
  //     (a, b) => new Date(b.lastModifiedTime) - new Date(a.lastModifiedTime)
  //   );

  //   if (sortedScenes.length > 0) {
  //     const sceneToSet = sortedScenes[0];
  //     if (
  //       !currentSample ||
  //       (autoLoad && sceneToSet.sceneName !== currentSample.sceneName)
  //     ) {
  //       setCurrentSample(sceneToSet);
  //       setMaskImage(null);
  //       updateImageSources(sceneToSet.sceneName);
  //       return;
  //     }

  //     const updatedScene = availableSamples.find(
  //       (scene) => scene.sceneName === currentSample.sceneName
  //     );

  //     if (!_.isEqual(updatedScene, currentSample)) {
  //       setCurrentSample(updatedScene);
  //       setMaskImage(null);
  //       updateImageSources(updatedScene.sceneName);
  //     }
  //   }
  // }, [availableSamples, currentSample, autoLoad]);

  const getAvailableTeams = async () => {
    try {
      const response = await axios.get("http://127.0.0.1:5000/api/teams");
      return response?.data?.teams ?? [];
    } catch (error) {
      console.error("Error fetching available team:", error);
      return [];
    }
  };


  const getAvailableScenes = async () => {
    try {
      return [];
      // const response = await axios.get("http://127.0.0.1:5000/api/scenes");
      // return response?.data?.scenes ?? [];
    } catch (error) {
      console.error("Error fetching available files:", error);
      return [];
    }
  };

  // const handleTeamSelection = (selectedOption) => {
  //   setCurrentTeam(selectedOption.value);
  // };


  // const handleSceneSelection = (selectedOption) => {
  //   setCurrentScene(selectedOption.value);
  //   handleClearManualAnnotations();
  //   setMaskImage(null);
  //   updateImageSources(selectedOption.value.sceneName);
  // };

  // const handleMouseDown = (e) => {
  //   if (annotationMode === "positive") {
  //     setPositiveKeypoints((points) => [
  //       ...points,
  //       e.target.getStage().getPointerPosition(),
  //     ]);
  //     return;
  //   }

  //   if (annotationMode === "negative") {
  //     setNegativeKeypoints((points) => [
  //       ...points,
  //       e.target.getStage().getPointerPosition(),
  //     ]);
  //     return;
  //   }

  //   setDrawing(true);
  //   const pos = e.target.getStage().getPointerPosition();
  //   setRect({ x: pos.x, y: pos.y, width: 10, height: 10 });
  // };

  // const handleMouseMove = (e) => {
  //   if (!drawing) return;
  //   const pos = e.target.getStage().getPointerPosition();
  //   setRect((rect) => ({
  //     ...rect,
  //     width: pos.x - rect.x,
  //     height: pos.y - rect.y,
  //   }));
  // };

  // const handleMouseUp = () => {
  //   setDrawing(false);
  // };

  // const annotate = (event) => {
  //   setAnnotationIsLoading(true);
  //   const { x, y, width, height } = rect;
  //   axios
  //     .post("http://127.0.0.1:5000/api/annotate", {
  //       bbox: {
  //         x1: x / image.scale,
  //         y1: y / image.scale,
  //         x2: (x + width) / image.scale,
  //         y2: (y + height) / image.scale,
  //       },
  //       positiveKeypoints: positiveKeypoints.map((pos) => [
  //         pos.x / image.scale,
  //         pos.y / image.scale,
  //       ]),
  //       negativeKeypoints: negativeKeypoints.map((pos) => [
  //         pos.x / image.scale,
  //         pos.y / image.scale,
  //       ]),
  //       outlierThreshold: outlierThreshold ?? currentResult?.outlierThreshold,
  //       sceneName: currentSample?.sceneName,
  //     })
  //     .then((response) => {
  //       updateAvailableData();
  //     })
  //     .catch((error) => {
  //       console.error("Error annotating:", error);
  //       console.error("Error fetching coordinates:", error);
  //     })
  //     .finally(() => {
  //       setAnnotationIsLoading(false);
  //     });
  // };

  // const currentResult = currentSample?.result;

  const annotationModeSelectOptions = ["positive", "negative", "box"].map(
    (mode) => ({
      value: mode,
      label: mode.charAt(0).toUpperCase() + mode.slice(1),
    })
  );

  // console.log("Current rect:", rect);
  // console.log("Current positive keypoints:", positiveKeypoints);
  // console.log("Current negative keypoints:", negativeKeypoints);

  return (
    <>
      <Modal
        isOpen={isLoading}
        contentLabel="Loading Modal"
        ariaHideApp={false}
        style={{
          overlay: {
            display: "flex",
            alignItems: "center",
            justifyContent: "center",
            backgroundColor: "rgba(0, 0, 0, 0.5)",
          },
          content: {
            position: "absolute",
            top: "50%",
            left: "50%",
            background: "none",
            border: "none",
            padding: "0",
          },
        }}
      >
        <ClipLoader color="#ffffff" loading={true} size={30} />
      </Modal>

      <header className="flex items-center justify-between p-4 bg-gray-800 text-white font-sans">
        <h1 className="text-4xl font-bold">👕👔🥼🦺🧥👘</h1>

        {/* <div className="flex items-center justify-between space-x-4">
          <div className="flex items-center space-x-4">
            <div>Manual label mode</div>
            <Select
              value={annotationModeSelectOptions.find(
                (option) => option.value === annotationMode
              )}
              onChange={(selectedOption) =>
                setAnnotationMode(selectedOption.value)
              }
              options={annotationModeSelectOptions}
              placeholder="Select annotation mode"
              styles={customStyles}
            />
            <button
              className="px-2 py-2 bg-blue-500 hover:bg-blue-700 text-white font-bold rounded"
              onClick={handleClearManualAnnotations}
            >
              Clear manual labels
            </button>
          </div>

          <div className="flex items-center space-x-4">
            <Switch
              onChange={() => {
                setShowMask(!showMask);
              }}
              checked={showMask}
            />
            <div>Show Mask</div>
            <Switch
              onChange={() => setShowDepth(!showDepth)}
              checked={showDepth}
            />
            <div>Show Depth</div>
          </div>
        </div> */}
      </header>


      <main className="flex justify-between">
        <div className="ml-1 mt-1">
          <SampleSelection
            currentTeam={currentTeam}
            setCurrentTeam={setCurrentTeam}
            currentScene={currentSample}
            setCurrentScene={setCurrentSample}
            availableTeams={availableTeams}
            availableScenes={availableSamples}
            handleRefresh={handleRefresh}
          />
        </div>

        {/* {!!image && <ImageDisplay
          image={image}
          maskImage={maskImage}
          depthImage={depthImage}
          rect={rect}
          setRect={setRect}
          positiveKeypoints={positiveKeypoints}
          setPositiveKeypoints={setPositiveKeypoints}
          negativeKeypoints={negativeKeypoints}
          setNegativeKeypoints={setNegativeKeypoints}
          annotationMode={annotationMode}
          showDepth={showDepth}
          showMask={showMask}
        />} */}
      </main>
    </>
  );
};

export default App;
