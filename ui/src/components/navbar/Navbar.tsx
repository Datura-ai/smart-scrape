import React from "react"
import css from "./Navbar.module.css";
import cn from "classnames";
import ThemeToggle from "../theme/ThemeToggle";

const Navbar : React.FC = ()=> {
    return(
        <nav className={cn(
            css[`navbarShadow`], `mb-3`)}>
            <div className="container mx-auto">
                <div className="flex items-center justify-between" >
                    <div className="w-22 py-2"><img src="/images/logo.png" alt="taotensor"  className={css.logo}  /></div>
                    <div className="flex">
                    <button className="mr-4">
                        <ThemeToggle />
                    </button>
                    <button
                        className="bg-gray-300 rounded-full hover:bg-gray-400 transform hover:scale-120 transition-transform duration-300 ease-in-out focus:outline-none focus:shadow-outline-gray-400 active:bg-gray-400"
                        >
                    </button>
                    </div>
                </div>
            </div>
        </nav>
    )
}

export default Navbar