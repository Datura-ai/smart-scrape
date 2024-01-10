import { atom } from "recoil";

const ThemeMode = atom({
    key: 'ThemeMode',
    default: {
      dark : (() => {
        const theme = localStorage.getItem('theme');
        console.log('Current theme:', theme);
        return theme === 'light' ? false : true;
      })()
    },
  });

export default ThemeMode;