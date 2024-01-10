const TickIcon = ({ ...props }) => {
    const {fillColor} = props
    return (
        <svg width="20px" height="20px" viewBox="0 0 1024 1024" xmlns="http://www.w3.org/2000/svg" fill={fillColor ? fillColor : "#000000"} stroke={fillColor ? fillColor : "#000000"}><g id="SVGRepo_bgCarrier" stroke-width="0"></g><g id="SVGRepo_tracerCarrier" stroke-linecap="round" stroke-linejoin="round"></g><g id="SVGRepo_iconCarrier"><path fill={fillColor ? fillColor : "#ffffff"} d="M704 192h160v736H160V192h160.064v64H704v-64zM311.616 537.28l-45.312 45.248L447.36 763.52l316.8-316.8-45.312-45.184L447.36 673.024 311.616 537.28zM384 192V96h256v96H384z"></path></g></svg>
    );
};

export default TickIcon;