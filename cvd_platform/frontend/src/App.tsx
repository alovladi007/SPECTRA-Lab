/**
 * CVD Platform Main Application
 * React + TypeScript + Material-UI
 */

import React, { useState, useEffect } from 'react';
import {
  AppBar,
  Box,
  CssBaseline,
  Drawer,
  IconButton,
  List,
  ListItem,
  ListItemButton,
  ListItemIcon,
  ListItemText,
  Toolbar,
  Typography,
  ThemeProvider,
  createTheme,
} from '@mui/material';
import {
  Menu as MenuIcon,
  Dashboard,
  Science,
  BarChart,
  Settings,
  Warning,
  Assessment,
} from '@mui/icons-material';

// Component imports
import DashboardView from './components/Dashboard';
import ProcessControlView from './components/ProcessControl';
import SPCView from './components/SPCCharts';
import AnalyticsView from './components/Analytics';
import RecipeManagement from './components/RecipeManagement';

const drawerWidth = 240;

// Dark theme configuration
const darkTheme = createTheme({
  palette: {
    mode: 'dark',
    primary: {
      main: '#2196f3',
    },
    secondary: {
      main: '#f50057',
    },
    background: {
      default: '#121212',
      paper: '#1e1e1e',
    },
  },
});

interface MenuItem {
  text: string;
  icon: React.ReactElement;
  view: string;
}

const menuItems: MenuItem[] = [
  { text: 'Dashboard', icon: <Dashboard />, view: 'dashboard' },
  { text: 'Process Control', icon: <Science />, view: 'control' },
  { text: 'SPC Charts', icon: <BarChart />, view: 'spc' },
  { text: 'Analytics', icon: <Assessment />, view: 'analytics' },
  { text: 'Recipes', icon: <Settings />, view: 'recipes' },
  { text: 'Alarms', icon: <Warning />, view: 'alarms' },
];

function App() {
  const [mobileOpen, setMobileOpen] = useState(false);
  const [currentView, setCurrentView] = useState('dashboard');
  const [equipmentStatus, setEquipmentStatus] = useState<any>(null);

  useEffect(() => {
    // Fetch equipment status on mount
    fetchEquipmentStatus();

    // Set up polling for real-time updates
    const interval = setInterval(fetchEquipmentStatus, 5000);

    return () => clearInterval(interval);
  }, []);

  const fetchEquipmentStatus = async () => {
    try {
      const response = await fetch('http://localhost:8000/equipment/status');
      const data = await response.json();
      setEquipmentStatus(data);
    } catch (error) {
      console.error('Failed to fetch equipment status:', error);
    }
  };

  const handleDrawerToggle = () => {
    setMobileOpen(!mobileOpen);
  };

  const handleMenuClick = (view: string) => {
    setCurrentView(view);
    setMobileOpen(false);
  };

  const renderView = () => {
    switch (currentView) {
      case 'dashboard':
        return <DashboardView />;
      case 'control':
        return <ProcessControlView />;
      case 'spc':
        return <SPCView />;
      case 'analytics':
        return <AnalyticsView />;
      case 'recipes':
        return <RecipeManagement />;
      default:
        return <DashboardView />;
    }
  };

  const drawer = (
    <div>
      <Toolbar>
        <Typography variant="h6" noWrap component="div">
          CVD Platform
        </Typography>
      </Toolbar>
      <List>
        {menuItems.map((item) => (
          <ListItem key={item.text} disablePadding>
            <ListItemButton
              selected={currentView === item.view}
              onClick={() => handleMenuClick(item.view)}
            >
              <ListItemIcon>{item.icon}</ListItemIcon>
              <ListItemText primary={item.text} />
            </ListItemButton>
          </ListItem>
        ))}
      </List>
    </div>
  );

  return (
    <ThemeProvider theme={darkTheme}>
      <Box sx={{ display: 'flex' }}>
        <CssBaseline />
        <AppBar
          position="fixed"
          sx={{
            width: { sm: `calc(100% - ${drawerWidth}px)` },
            ml: { sm: `${drawerWidth}px` },
          }}
        >
          <Toolbar>
            <IconButton
              color="inherit"
              aria-label="open drawer"
              edge="start"
              onClick={handleDrawerToggle}
              sx={{ mr: 2, display: { sm: 'none' } }}
            >
              <MenuIcon />
            </IconButton>
            <Typography variant="h6" noWrap component="div" sx={{ flexGrow: 1 }}>
              {menuItems.find((item) => item.view === currentView)?.text || 'CVD Platform'}
            </Typography>
            {equipmentStatus && (
              <Box sx={{ display: 'flex', alignItems: 'center', gap: 2 }}>
                <Typography variant="body2">
                  Equipment: {equipmentStatus.equipment_id}
                </Typography>
                <Typography variant="body2">
                  Status: {equipmentStatus.status}
                </Typography>
                <Typography variant="body2">
                  Wafer: {equipmentStatus.current_wafer}
                </Typography>
              </Box>
            )}
          </Toolbar>
        </AppBar>
        <Box
          component="nav"
          sx={{ width: { sm: drawerWidth }, flexShrink: { sm: 0 } }}
        >
          <Drawer
            variant="temporary"
            open={mobileOpen}
            onClose={handleDrawerToggle}
            ModalProps={{
              keepMounted: true,
            }}
            sx={{
              display: { xs: 'block', sm: 'none' },
              '& .MuiDrawer-paper': {
                boxSizing: 'border-box',
                width: drawerWidth,
              },
            }}
          >
            {drawer}
          </Drawer>
          <Drawer
            variant="permanent"
            sx={{
              display: { xs: 'none', sm: 'block' },
              '& .MuiDrawer-paper': {
                boxSizing: 'border-box',
                width: drawerWidth,
              },
            }}
            open
          >
            {drawer}
          </Drawer>
        </Box>
        <Box
          component="main"
          sx={{
            flexGrow: 1,
            p: 3,
            width: { sm: `calc(100% - ${drawerWidth}px)` },
          }}
        >
          <Toolbar />
          {renderView()}
        </Box>
      </Box>
    </ThemeProvider>
  );
}

export default App;
