(() => {
  "use strict";

  const state = {
    viewer: null,
    clouds: new Map(),
    viewMode: "original",
    materialMode: "rgb",
  };

  const status = (text) => {
    const el = document.getElementById("viewerStatus");
    if (el) el.textContent = text;
  };

  function defineClassificationColors(viewer) {
    const definitions = {
      0: { visible: true, name: "Nunca classificado", color: [0.55, 0.55, 0.55, 1.0] },
      1: { visible: true, name: "Não-solo", color: [0.92, 0.55, 0.18, 1.0] },
      2: { visible: true, name: "Solo", color: [0.28, 0.72, 0.30, 1.0] },
      7: { visible: true, name: "Ruído", color: [1.0, 0.0, 1.0, 1.0] },
    };
    Object.keys(definitions).forEach((key) => {
      viewer.classifications[key] = definitions[key];
    });
    if (!viewer.classifications.DEFAULT) {
      viewer.classifications.DEFAULT = {
        visible: true,
        name: "default",
        color: [0.35, 0.55, 0.70, 0.7],
      };
    }
  }

  function init() {
    if (typeof Potree === "undefined") {
      status("Potree não encontrado no pacote.");
      return;
    }

    const viewer = new Potree.Viewer(
      document.getElementById("potree_render_area")
    );
    viewer.setEDLEnabled(true);
    viewer.setFOV(60);
    viewer.setPointBudget(7500000);
    viewer.setBackground("black");
    viewer.setMinNodeSize(30);

    if (viewer.orbitControls) {
      viewer.setControls(viewer.orbitControls);
      if ("rotationSpeed" in viewer.orbitControls) {
        viewer.orbitControls.rotationSpeed = 6.0;
      }
      if ("fadeFactor" in viewer.orbitControls) {
        viewer.orbitControls.fadeFactor = 18.0;
      }
    } else if (viewer.earthControls) {
      viewer.setControls(viewer.earthControls);
    }

    defineClassificationColors(viewer);
    state.viewer = viewer;

    viewer.loadGUI(() => {
      viewer.setLanguage("en");
      const sidebar = document.getElementById(
        "potree_sidebar_container"
      );
      if (sidebar) sidebar.style.display = "none";
    });

    status("Viewport pronta · Potree · LOD dinâmico");
  }

  function configurePointcloud(pointcloud, classified) {
    const material = pointcloud.material;
    if (!material) return;
    material.size = 1.1;
    material.minSize = 1;
    material.maxSize = 12;
    material.pointSizeType =
      Potree.PointSizeType.ATTENUATED;
    material.shape = Potree.PointShape.CIRCLE;
    material.opacity = 1.0;
    material.activeAttributeName =
      classified ? "classification" : "rgba";
  }

  function removeCloud(key) {
    const current = state.clouds.get(key);
    if (!current || !state.viewer) return;
    try {
      if (
        typeof state.viewer.scene.removePointCloud
          === "function"
      ) {
        state.viewer.scene.removePointCloud(
          current.pointcloud
        );
      } else {
        current.pointcloud.visible = false;
      }
    } catch (_) {
      current.pointcloud.visible = false;
    }
    state.clouds.delete(key);
  }

  function applyViewMode() {
    for (const [key, entry] of state.clouds.entries()) {
      if (state.viewMode === "both") {
        entry.pointcloud.visible = true;
      } else {
        entry.pointcloud.visible =
          key === state.viewMode;
      }
    }
  }

  function setMaterialMode(mode) {
    state.materialMode = mode;
    for (const entry of state.clouds.values()) {
      const material = entry.pointcloud.material;
      if (!material) continue;
      if (mode === "classification") {
        material.activeAttributeName =
          "classification";
      } else if (mode === "elevation") {
        material.activeAttributeName =
          "elevation";
      } else {
        material.activeAttributeName = "rgba";
      }
    }
  }

  function loadCloud(
    key,
    url,
    name,
    classified,
    activate
  ) {
    if (!state.viewer) {
      status("Viewer ainda não inicializado.");
      return;
    }
    removeCloud(key);
    status("A carregar " + name + "…");

    Potree.loadPointCloud(url, name, (event) => {
      if (!event || !event.pointcloud) {
        status("Falha ao carregar " + name);
        return;
      }
      const pointcloud = event.pointcloud;
      configurePointcloud(pointcloud, classified);
      state.viewer.scene.addPointCloud(pointcloud);
      state.clouds.set(
        key,
        { pointcloud, classified, name }
      );

      if (activate) {
        state.viewMode = key;
      }
      applyViewMode();
      if (classified) {
        setMaterialMode("classification");
      } else if (
        state.materialMode === "classification"
      ) {
        setMaterialMode("rgb");
      }

      try {
        state.viewer.fitToScreen(0.8);
      } catch (_) {}
      window.requestAnimationFrame(() => {
        try {
          state.viewer.fitToScreen(0.8);
        } catch (_) {}
      });
      window.setTimeout(() => {
        try {
          state.viewer.fitToScreen(0.8);
        } catch (_) {}
      }, 350);
      status(name + " · LOD dinâmico");
    });
  }

  function setViewMode(mode) {
    if (
      !["original", "classified", "both"].includes(
        mode
      )
    ) {
      return;
    }
    state.viewMode = mode;
    applyViewMode();
    if (mode === "classified") {
      setMaterialMode("classification");
    }
  }

  function fit() {
    if (!state.viewer) return;
    try {
      state.viewer.fitToScreen(0.8);
    } catch (_) {}
  }

  function clear() {
    Array.from(state.clouds.keys()).forEach(
      removeCloud
    );
    state.viewMode = "original";
    state.materialMode = "rgb";
    status("Abra uma nuvem LAS / LAZ");
  }

  window.LASViewer = {
    loadCloud,
    setViewMode,
    setMaterialMode,
    fit,
    clear,
  };

  init();
})();
