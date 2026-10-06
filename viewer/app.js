(() => {
  "use strict";

  const state = {
    viewer: null,
    clouds: new Map(),
    viewMode: "original",
    materialMode: "rgb",
    autoFitted: false,
    mdtMesh: null,
    mdtPreview: null,
    mdtColorMode: "elevation",
  };

  const status = (text) => {
    const el = document.getElementById("viewerStatus");
    if (el) el.textContent = text;
    console.log("[LASViewer]", text);
  };

  function getAttribute(pointcloud, name) {
    try {
      if (typeof pointcloud.getAttribute === "function") {
        return pointcloud.getAttribute(name);
      }
    } catch (_) {}
    return null;
  }

  function rgbAttributeName(pointcloud) {
    if (getAttribute(pointcloud, "rgba")) return "rgba";
    if (getAttribute(pointcloud, "rgb")) return "rgb";
    return null;
  }

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

    material.size = 1.4;
    material.minSize = 1;
    material.maxSize = 14;
    material.pointSizeType =
      Potree.PointSizeType.ATTENUATED;
    material.shape = Potree.PointShape.CIRCLE;
    material.opacity = 1.0;

    const rgb = rgbAttributeName(pointcloud);
    if (classified) {
      material.activeAttributeName = "classification";
    } else if (rgb) {
      material.activeAttributeName = rgb;
    } else {
      material.activeAttributeName = "elevation";
      state.materialMode = "elevation";
    }

    console.log("[LASViewer] cloud attributes", {
      rgb: rgb,
      classification: Boolean(
        getAttribute(pointcloud, "classification")
      ),
      active: material.activeAttributeName,
    });
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

  function removeMDT() {
    if (!state.mdtMesh || !state.viewer) return;
    const mesh = state.mdtMesh;
    state.viewer.scene.scene.remove(mesh);
    mesh.geometry.dispose();
    mesh.material.dispose();
    state.mdtMesh = null;
    state.mdtPreview = null;
  }

  function buildMDTGeometry(data) {
    if (typeof THREE === "undefined") {
      throw new Error("Three.js não está disponível no Potree");
    }
    const nx = data.width;
    const ny = data.height;
    if (nx < 2 || ny < 2 || data.z.length !== nx * ny ||
        data.state.length !== nx * ny || nx * ny > 350000) {
      throw new Error("Dimensões inválidas na pré-visualização MDT");
    }
    const valid = data.z.filter((v) => v !== null && Number.isFinite(v));
    if (!valid.length) throw new Error("MDT sem células válidas");
    let minZ = Infinity;
    let maxZ = -Infinity;
    for (const value of valid) {
      if (value < minZ) minZ = value;
      if (value > maxZ) maxZ = value;
    }
    const positions = new Float32Array(nx * ny * 3);
    const colors = new Float32Array(nx * ny * 3);
    const cell = data.resolution_m;

    for (let r = 0; r < ny; r++) {
      for (let c = 0; c < nx; c++) {
        const i = r * nx + c;
        const z = data.z[i];
        positions[i * 3] = c * cell;
        positions[i * 3 + 1] = -r * cell;
        positions[i * 3 + 2] = z === null ? 0 : z - minZ;
      }
    }
    const triangles = [];
    for (let r = 0; r + 1 < ny; r++) {
      for (let c = 0; c + 1 < nx; c++) {
        const a = r * nx + c;
        const b = a + 1;
        const d = a + nx;
        const e = d + 1;
        // Never bridge nodata; missing Ground stays a real hole in the mesh.
        if (data.state[a] > 0 && data.state[b] > 0 &&
            data.state[d] > 0 && data.z[a] !== null &&
            data.z[b] !== null && data.z[d] !== null) {
          triangles.push(a, d, b);
        }
        if (data.state[b] > 0 && data.state[d] > 0 &&
            data.state[e] > 0 && data.z[b] !== null &&
            data.z[d] !== null && data.z[e] !== null) {
          triangles.push(b, d, e);
        }
      }
    }
    const geometry = new THREE.BufferGeometry();
    const positionAttribute = new THREE.BufferAttribute(positions, 3);
    const colorAttribute = new THREE.BufferAttribute(colors, 3);
    if (typeof geometry.setAttribute === "function") {
      geometry.setAttribute("position", positionAttribute);
      geometry.setAttribute("color", colorAttribute);
    } else {
      geometry.addAttribute("position", positionAttribute);
      geometry.addAttribute("color", colorAttribute);
    }
    geometry.setIndex(triangles);
    geometry.computeVertexNormals();
    const mesh = new THREE.Mesh(
      geometry,
      new THREE.MeshBasicMaterial({
        vertexColors: true,
        side: THREE.DoubleSide,
        depthWrite: true,
      })
    );
    mesh.position.set(data.xmin, data.ymax, minZ);
    mesh.frustumCulled = false;
    mesh.userData.mdtMin = minZ;
    mesh.userData.mdtMax = maxZ;
    return mesh;
  }

  function setMDTColorMode(mode) {
    if (!state.mdtMesh || !state.mdtPreview) return;
    if (!["elevation", "hillshade", "observation"].includes(mode)) return;
    const data = state.mdtPreview;
    const colors = state.mdtMesh.geometry.getAttribute("color");
    const minZ = state.mdtMesh.userData.mdtMin;
    const maxZ = state.mdtMesh.userData.mdtMax;
    const nx = data.width;
    const ny = data.height;

    const zAt = (r, c, defaultZ) => {
      const i = Math.min(ny - 1, Math.max(0, r)) * nx +
                Math.min(nx - 1, Math.max(0, c));
      return data.z[i] === null ? defaultZ : data.z[i];
    };
    for (let r = 0; r < ny; r++) {
      for (let c = 0; c < nx; c++) {
        const i = r * nx + c;
        const z = data.z[i];
        if (z === null) {
          colors.setXYZ(i, 0, 0, 0);
          continue;
        }
        if (mode === "observation") {
          if (data.state[i] === 1) colors.setXYZ(i, 0.20, 0.74, 0.33);
          else colors.setXYZ(i, 0.95, 0.68, 0.22);
        } else if (mode === "hillshade") {
          const dx = (zAt(r, c + 1, z) - zAt(r, c - 1, z)) / (2 * data.resolution_m);
          const dy = (zAt(r - 1, c, z) - zAt(r + 1, c, z)) / (2 * data.resolution_m);
          const mag = Math.sqrt(dx * dx + dy * dy + 1);
          const light = Math.max(0.08, (0.5 * -dx + 0.4 * -dy + 0.77) / mag);
          colors.setXYZ(i, light, light, light);
        } else {
          const t = Math.max(0, Math.min(1, (z - minZ) / Math.max(0.001, maxZ - minZ)));
          colors.setXYZ(i, 0.14 + 0.7 * t, 0.34 + 0.45 * t, 0.74 - 0.57 * t);
        }
      }
    }
    colors.needsUpdate = true;
    state.mdtColorMode = mode;
    status("MDT R20.5 · " + mode + " · verde=medido / amarelo=interpolado na vista Observação");
  }

  function showMDTPreview(data) {
    if (!state.viewer) throw new Error("Viewer ainda não inicializado");
    removeMDT();
    state.mdtPreview = data;
    state.mdtMesh = buildMDTGeometry(data);
    state.viewer.scene.scene.add(state.mdtMesh);
    setMDTColorMode("elevation");
    setViewMode("mdt");
    status("MDT R20.5 calculado em memória · pré-visualização 3D · exportação pendente");
  }

  function applyViewMode() {
    if (state.mdtMesh) state.mdtMesh.visible = state.viewMode === "mdt";
    for (const [key, entry] of state.clouds.entries()) {
      if (state.viewMode === "both") {
        // Do not overlay the *inferred* mantle onto measured Ground by default.
        entry.pointcloud.visible = key === "original" || key === "classified";
      } else {
        entry.pointcloud.visible =
          key === state.viewMode;
      }
    }
  }

  function setMaterialMode(mode) {
    state.materialMode = mode;
    for (const entry of state.clouds.values()) {
      const pc = entry.pointcloud;
      const material = pc.material;
      if (!material) continue;

      if (mode === "classification") {
        material.activeAttributeName =
          getAttribute(pc, "classification")
            ? "classification"
            : "elevation";
      } else if (mode === "elevation") {
        material.activeAttributeName = "elevation";
      } else {
        material.activeAttributeName =
          rgbAttributeName(pc) || "elevation";
      }
    }
  }

  function fitRepeatedly() {
    const delays = [0, 100, 350, 900, 1800];
    for (const delay of delays) {
      window.setTimeout(() => {
        try {
          state.viewer.fitToScreen(0.8);
        } catch (error) {
          console.warn("[LASViewer] fit failed", error);
        }
      }, delay);
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
      try {
        if (!event || !event.pointcloud) {
          status("Falha ao carregar " + name);
          return;
        }

        const pointcloud = event.pointcloud;
        configurePointcloud(pointcloud, classified);
        pointcloud.visible = true;
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
        } else {
          setMaterialMode("rgb");
        }

        if (!state.autoFitted) {
          fitRepeatedly();
          state.autoFitted = true;
        }
        status(name + " · LOD dinâmico · pontos a carregar");
      } catch (error) {
        console.error("[LASViewer] loadCloud failed", error);
        status("Erro na viewport: " + String(error));
      }
    });
  }

  function setViewMode(mode) {
    if (
      !["original", "classified", "mantle", "both", "mdt"].includes(
        mode
      )
    ) {
      return;
    }
    state.viewMode = mode;
    applyViewMode();
    if (mode === "classified") {
      setMaterialMode("classification");
    } else if (mode === "mantle") {
      // MantleState is color-coded through RGB; LAS class 0 is deliberate.
      setMaterialMode("rgb");
    }

    // Switching Original / Final Ground / Compare / Manto never moves the camera.
    // The same eye position, target, rotation and zoom are kept so differences
    // are inspected at exactly the same location.
  }

  function fit() {
    if (!state.viewer) return;
    fitRepeatedly();
  }

  function clear() {
    removeMDT();
    Array.from(state.clouds.keys()).forEach(
      removeCloud
    );
    state.viewMode = "original";
    state.materialMode = "rgb";
    state.autoFitted = false;
    status("Abra uma nuvem LAS / LAZ");
  }

  window.addEventListener("error", (event) => {
    console.error(
      "[LASViewer] window error",
      event.message,
      event.error
    );
  });

  window.addEventListener(
    "unhandledrejection",
    (event) => {
      console.error(
        "[LASViewer] promise rejection",
        event.reason
      );
    }
  );

  window.LASViewer = {
    loadCloud,
    showMDTPreview,
    clearMDTPreview: removeMDT,
    setMDTColorMode,
    setViewMode,
    setMaterialMode,
    fit,
    clear,
  };

  init();
})();
