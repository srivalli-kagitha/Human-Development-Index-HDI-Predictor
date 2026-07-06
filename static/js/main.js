// --- Human Development Index (HDI) Predictor JS Functionality ---

document.addEventListener("DOMContentLoaded", function () {
  // 1. Theme Management (Dark / Light Mode)
  const themeToggleBtn = document.getElementById("themeToggle");
  const themeIcon = document.getElementById("themeIcon");
  
  // Retrieve saved theme or default to system preference
  const savedTheme = localStorage.getItem("hdi-theme");
  const systemPrefersDark = window.matchMedia("(prefers-color-scheme: dark)").matches;
  const initialTheme = savedTheme || (systemPrefersDark ? "dark" : "light");
  
  // Set theme attribute
  document.documentElement.setAttribute("data-theme", initialTheme);
  updateThemeIcon(initialTheme);
  
  if (themeToggleBtn) {
    themeToggleBtn.addEventListener("click", function () {
      const currentTheme = document.documentElement.getAttribute("data-theme");
      const newTheme = currentTheme === "dark" ? "light" : "dark";
      
      document.documentElement.setAttribute("data-theme", newTheme);
      localStorage.setItem("hdi-theme", newTheme);
      updateThemeIcon(newTheme);
    });
  }
  
  function updateThemeIcon(theme) {
    if (!themeIcon) return;
    if (theme === "dark") {
      themeIcon.className = "bi bi-sun-fill";
      if (themeToggleBtn) themeToggleBtn.title = "Switch to Light Mode";
    } else {
      themeIcon.className = "bi bi-moon-fill";
      if (themeToggleBtn) themeToggleBtn.title = "Switch to Dark Mode";
    }
  }

  // 2. Synchronize Range Sliders and Numeric Inputs
  const inputSyncs = [
    { numId: "life_expectancy", sliderId: "life_expectancy_slider" },
    { numId: "expected_schooling", sliderId: "expected_schooling_slider" },
    { numId: "mean_schooling", sliderId: "mean_schooling_slider" },
    { numId: "gni_per_capita", sliderId: "gni_per_capita_slider" }
  ];

  inputSyncs.forEach(pair => {
    const numInput = document.getElementById(pair.numId);
    const sliderInput = document.getElementById(pair.sliderId);

    if (numInput && sliderInput) {
      // Sync slider value to text input
      sliderInput.addEventListener("input", function () {
        numInput.value = this.value;
        validateSchoolingRelation();
      });

      // Sync text input value to slider
      numInput.addEventListener("input", function () {
        sliderInput.value = this.value;
        validateSchoolingRelation();
      });
    }
  });

  // Relation validation helper: Mean Schooling <= Expected Schooling
  function validateSchoolingRelation() {
    const eysInput = document.getElementById("expected_schooling");
    const mysInput = document.getElementById("mean_schooling");
    const relationFeedback = document.getElementById("schooling-relation-feedback");

    if (eysInput && mysInput && relationFeedback) {
      const eys = parseFloat(eysInput.value) || 0;
      const mys = parseFloat(mysInput.value) || 0;

      if (mys > eys) {
        relationFeedback.style.display = "block";
        mysInput.classList.add("is-invalid");
      } else {
        relationFeedback.style.display = "none";
        mysInput.classList.remove("is-invalid");
      }
    }
  }

  // 3. Client-Side Validation on Submit
  const predictForm = document.getElementById("predictForm");
  if (predictForm) {
    predictForm.addEventListener("submit", function (e) {
      let isValid = true;
      
      const lifeExp = parseFloat(document.getElementById("life_expectancy").value);
      const eys = parseFloat(document.getElementById("expected_schooling").value);
      const mys = parseFloat(document.getElementById("mean_schooling").value);
      const gni = parseFloat(document.getElementById("gni_per_capita").value);
      
      // Range verification
      if (isNaN(lifeExp) || lifeExp < 20.0 || lifeExp > 100.0) {
        markInvalid("life_expectancy");
        isValid = false;
      } else {
        markValid("life_expectancy");
      }
      
      if (isNaN(eys) || eys < 0.0 || eys > 25.0) {
        markInvalid("expected_schooling");
        isValid = false;
      } else {
        markValid("expected_schooling");
      }
      
      if (isNaN(mys) || mys < 0.0 || mys > 20.0) {
        markInvalid("mean_schooling");
        isValid = false;
      } else {
        markValid("mean_schooling");
      }
      
      if (isNaN(gni) || gni < 100.0 || gni > 150000.0) {
        markInvalid("gni_per_capita");
        isValid = false;
      } else {
        markValid("gni_per_capita");
      }
      
      if (mys > eys) {
        markInvalid("mean_schooling");
        isValid = false;
      }
      
      if (!isValid) {
        e.preventDefault(); // Stop submission
        // Focus first invalid element
        const firstInvalid = document.querySelector(".is-invalid");
        if (firstInvalid) firstInvalid.focus();
      } else {
        // Show loading overlay during mock pipeline delay
        const loadingOverlay = document.getElementById("loadingOverlay");
        if (loadingOverlay) {
          loadingOverlay.style.display = "flex";
        }
      }
    });
  }

  function markInvalid(id) {
    const input = document.getElementById(id);
    if (input) input.classList.add("is-invalid");
  }

  function markValid(id) {
    const input = document.getElementById(id);
    if (input) input.classList.remove("is-invalid");
  }

  // 4. Download PDF prediction report
  const downloadPdfBtn = document.getElementById("downloadPdfBtn");
  if (downloadPdfBtn) {
    downloadPdfBtn.addEventListener("click", function () {
      window.print();
    });
  }

  // Bootstrap tooltips activation (if bootstrap is loaded)
  if (typeof bootstrap !== 'undefined' && bootstrap.Tooltip) {
    const tooltipTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="tooltip"]'));
    tooltipTriggerList.map(function (tooltipTriggerEl) {
      return new bootstrap.Tooltip(tooltipTriggerEl);
    });
  }
});
