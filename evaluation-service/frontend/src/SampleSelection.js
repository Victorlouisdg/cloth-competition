import React from "react";
import Select from "react-select";

const SampleSelection = ({
    currentTeam,
    setCurrentTeam,
    currentSample,
    setCurrentSample,
    availableTeams,
    availableScenes,
    handleRefresh,
    customStyles  // Import your customStyles object if you have it in a separate file
}) => {

    const teamSelectOptions = availableTeams.map((team) => ({
        value: team,
        label: team,
    }));

    const sceneSelectOptions = availableScenes
        .map((scene) => ({
            value: scene,
            label: scene.sceneName,
        }))
        .sort((a, b) => a.label.localeCompare(b.label));

    var observation_name = `http://127.0.0.1:5000/static/data/remote_dry_run/${currentTeam}`


    return (
        <div className="ml-5 mt-5">
            <h2 className="text-2xl font-bold">Sample selection</h2>

            <div className="flex justify-left mt-3">
                <div className="mr-2 mt-2">Team</div>
                <Select
                    value={teamSelectOptions.find(
                        (option) => option.value === currentTeam
                    )}
                    onChange={(selectedOption) => setCurrentTeam(selectedOption.value)}
                    options={teamSelectOptions}
                    placeholder="Select team"
                    styles={customStyles}
                />
            </div>

            <div className="flex justify-left mt-1">
                <div className="mr-2 mt-2">Sample</div>
                <Select
                    value={sceneSelectOptions.find(
                        (option) => option.value.sceneName === currentSample?.sceneName
                    )}
                    onChange={(selectedOption) => setCurrentSample(selectedOption.value)}
                    options={sceneSelectOptions}
                    placeholder="Sample"
                    styles={customStyles}
                />
            </div>

            <button
                className="mt-2 mb-2 px-3 py-1 bg-blue-500 hover:bg-blue-700 text-white font-bold rounded"
                onClick={handleRefresh}
            >
                Refresh samples
            </button>

            <p>Selected observation:</p>
            <p style={{ wordBreak: 'break-word' }}>{observation_name}</p>
        </div>
    );
};

export default SampleSelection;