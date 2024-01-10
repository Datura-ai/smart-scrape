import React from 'react'

const NotFound : React.FC = ()=> {
    return (
      <div className="flex items-center justify-center h-screen bg-gray-100">
        <div className="text-center">
          <div><img src="/images/404.png" /></div>
          <p className="text-gray-400 mb-4">The page you're looking for doesn't exist.</p>
        </div>
      </div>
    )
}

export default NotFound