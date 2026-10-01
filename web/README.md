# Website

React + Vite site for the fleet plate check demo. No backend: the model results are
exported once by `scripts/export_demo.py` into `public/demo/`, and the card check runs
in the browser (`src/verify.js`, same rules as `scripts/fleet_check.py`).

```
npm install
npm run dev      # local preview at http://localhost:5173
npm run build    # production build in dist/
```
