import { EyeVisibleIcon, EyeInVisibleIcon } from "../icons/index";
import cn from "classnames";
import React, { InputHTMLAttributes, useState } from "react";
import css from "./InputComponent.module.css";

export interface InputProps extends InputHTMLAttributes<HTMLInputElement> {
  inputClassName?: string;
  labelClassName?: string;
  containerClassName?: string;
  icon?: React.ReactElement;
  labelText?: string;
  error?: string;
  showErrorMessage?: boolean;
  autocomplete?: string;
}

const InputComponent: React.FC<InputProps> = (props) => {
  const {
    inputClassName,
    labelClassName,
    icon,
    type = "text",
    labelText,
    error,
    className,
    containerClassName,
    name,
    showErrorMessage,
    ...rest
  } = props;
  const [inputType, setInputType] = useState<string | undefined>(type);
  const handleToggleInputType = () => {
    setInputType(inputType === "password" ? "text" : "password");
  };

  return (
    <div
      className={`${error && "error"} relative z-0 w-full ${
        containerClassName || ""
      }`}
    >
      <input
        name={name}
        placeholder=" "
        {...rest}
        type={inputType}
        className={cn(
          inputClassName,
          className,
          css[`input-component`],
          css[`${error ? "error-style" : "input-style"}`],
          `p-3 block w-full rounded-lg mt-0 bg-transparent appearance-none shadow-md shadow-outline-normal h-10 hover:border-c-gray`
        )}
      />

      <label
        htmlFor={type}
        className={cn(
          css["input-label"],
          css["label"],
          `absolute duration-300 px-4 top-2 -z-1 bg-regular origin-0 text-primary`,
          labelClassName
        )}
      >
        {labelText}
      </label>
      <span
        className={`absolute right-4 z-1 translate-y-[-100%] top-[50%] pr-3.5 w-3 h-3  text-primary`}
      >
        {icon}
      </span>
      {type === "password" && (
        <span
          className={`absolute z-0 cursor-pointer right-0 top-3 px-3 bottom-0  text-primary`}
        >
          {inputType === "password" ? (
            <EyeVisibleIcon handleClick={handleToggleInputType} />
          ) : (
            <EyeInVisibleIcon handleClick={handleToggleInputType} />
          )}
        </span>
      )}
      {error && showErrorMessage && (
        <span className={cn(css["error-message"], "text-red")} id="error">
          {error}
        </span>
      )}
    </div>
  );
};
export { InputComponent };
