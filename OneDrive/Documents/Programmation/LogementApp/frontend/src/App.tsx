import './App.css'
import { BrowserRouter, Route, Routes, Link, useNavigate } from 'react-router-dom'
import { useState, useEffect } from 'react'
import AddHouse from './AddHouse'
import BatimentForm from './BatimentForm'
import BatimentPage from './BatimentPage'
import Signup from './Signup'
import Login from './Login'
import RechercheBatiments from './RechercheBatiments'
import BatimentsList from './BatimentsList'
import ReviewPage from './Review'

function AppWrapper() {
  // On utilise un wrapper pour pouvoir utiliser useNavigate
  return (
    <BrowserRouter>
      <App />
    </BrowserRouter>
  )
}

function App() {
  const [userRole, setUserRole] = useState(null) // "voyageur" ou "hote"
  const [isLoggedIn, setIsLoggedIn] = useState(false)
  const navigate = useNavigate()

  useEffect(() => {
    const role = localStorage.getItem('role')
    const idClient = localStorage.getItem('id_client')
    if (idClient) {
      setIsLoggedIn(true)
      setUserRole(role)
    } else {
      setIsLoggedIn(false)
      setUserRole(null)
    }
  }, [])

  const handleLogout = () => {
    localStorage.removeItem('id_client')
    localStorage.removeItem('role')
    setIsLoggedIn(false)
    setUserRole(null)
    navigate('/Login')
  }

  return (
    <div>
      <div className="navbar">
        {isLoggedIn ? (
          <>
            <Link to="/RechercheBatiments">Recherche</Link>
            <Link to="/BatimentPage">BatimentPage</Link>
            <Link to="/review/1">ReviewPage</Link>
            {userRole === 'hote' && (
              <>
                <Link to="/BatimentForm">BatimentForm</Link>
                <Link to="/BatimentsList">Mes Batiments</Link>
                <Link to="/ajouterBatiment">Ajouter Batiment</Link>
              </>
            )}
            <button onClick={handleLogout}>Déconnexion</button>
          </>
        ) : (
          <>
            <Link to="/Login">Login</Link>
            <Link to="/Signup">Signup</Link>
          </>
        )}
      </div>

      <Routes>
        <Route path='/Login' element={<Login />} />
        <Route path='/Signup' element={<Signup />} />
        <Route path='/ajouterBatiment' element={<AddHouse />} />
        <Route path='/BatimentForm/:id?' element={<BatimentForm />} />
        <Route path='/RechercheBatiments' element={<RechercheBatiments />} />
        <Route path='/Batiments/:idBatiment' element={<BatimentPage />} />
        <Route path='/BatimentsList' element={<BatimentsList />} />
        <Route path="/review/:idProprio" element={<ReviewPage />} />
      </Routes>
    </div>
  )
}

export default AppWrapper
