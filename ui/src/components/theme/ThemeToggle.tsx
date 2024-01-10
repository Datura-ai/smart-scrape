import React from "react";
import { Moon, Sun } from "../icons";
import "./ThemeToggle.css";
import ThemeMode from "./ThemeMode";
import { useRecoilState } from "recoil";

const ThemeToggle = () => {
    const [mode,setDark] = useRecoilState(ThemeMode)
    const handleThemeChange = ()=> {
        setDark((prevMode:any) => {
            const newMode = { ...prevMode, dark: !prevMode.dark };
            if (newMode.dark) {
              document.documentElement.classList.add("dark");
              localStorage.setItem("theme", "dark");
            } else {
              document.documentElement.classList.remove("dark");
              localStorage.setItem("theme", "light");
            }
            return newMode;
          });
    }
    return (
      <div className='dark_mode_label' onClick={handleThemeChange}>
          { mode && mode.dark === true ? <Sun /> :
          <Moon />}
      </div>
    );
};

export default ThemeToggle;
