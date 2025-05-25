% Modell erstellen
model = createpde();
geometryFromEdges(model, @circleg);


f = @(location, state) sin(location.x.^2 + location.y.^2).^2;


% Koeffizienten setzen
specifyCoefficients(model, "m", 0, "d", 0, "c", 1, "a", 0, "f",f);

% Randbedingung (u = 0 auf dem Rand)
applyBoundaryCondition(model, "dirichlet", "Edge", 1:model.Geometry.NumEdges, "u", 0);

% Mesh generieren
generateMesh(model, "Hmax", 0.1);

% Lösen
results = solvepde(model);

% Plot
pdeplot(model, "XYData", results.NodalSolution);
title("Numerische Lösung von -\Delta u = sin(x^2 + y^2)");