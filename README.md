### Scratch Environment
WIP. Research purposes primarily. Only a scratch RL environment now：

1. No visual information for agent.
  
2. Agent can only perform standing long throw of smoke grenades.
  
3. Single-threaded. 


### Prerequisites

1. Copy "seeknade.cfg" and "gamestate_intergration_nadeseeker.cfg" to "cs2_path/game/csgo/cfg". The first file contians necessary console commands and the second file enables GSI. 

2. Add launch options "-condebug -conclearlog -insecure" to CS2 from Steam library. Their usages are a) output console logs, b) clear log file on game's launching, c) disable anti-cheat engine (just in case). 

3. In the game, select the practice gamemdoe and checks all "Infinite ***" options on the left sidebar. 

4. After entering the map, input "exec seeknade" in the game's "developer console". About the console, see https://prosettings.net/blog/open-console-cs2/


### Usage

.ini files in "./playground" are experiment scenes for agent training. 
The paper's experiment scenes are prepared (Need manually adding CS2 game folder to each .ini file first). 
The map is "Italy". 

Run "train.py" to start training. Important infos are printed such as "Press 0 to start". 

This is also a source code corresponding to a technical paper submitted to IEEE CoG 2024. 
