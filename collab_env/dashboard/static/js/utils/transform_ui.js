/**
 * Transform UI - Builds UI controls for TrackTransformer
 *
 * Creates standardized HTML controls for manual track alignment
 * and binds them to a TrackTransformer instance.
 */

export class TransformUI {
    constructor(transformer, container) {
        this.transformer = transformer;
        this.container = container;

        // UI elements
        this.enableCheckbox = null;
        this.debugCheckbox = null;
        this.debugDisplay = null;
        this.sliders = {};
        this.valueDisplays = {};

        this._createControls();
        this._bindEvents();
    }

    /**
     * Create all UI controls
     */
    _createControls() {
        const html = `
            <div class="transform-header">
                <label>Manual Track Alignment</label>
                <div class="transform-toggle-group">
                    <label class="transform-toggle">
                        <input type="checkbox" id="enableTransform">
                        Enable Manual Mode
                    </label>
                    <label class="transform-toggle">
                        <input type="checkbox" id="showDebugTransform">
                        Show Debug Info
                    </label>
                </div>
            </div>

            <div class="transform-section">
                <div class="transform-section-title">Translation (m)</div>
                <div class="transform-slider-row">
                    <label>X:</label>
                    <input type="range" id="translateX" min="-5.0" max="5.0" step="0.001" value="0">
                    <span class="transform-value" id="translateX-value">0.000</span>
                </div>
                <div class="transform-slider-row">
                    <label>Y:</label>
                    <input type="range" id="translateY" min="-5.0" max="5.0" step="0.001" value="0">
                    <span class="transform-value" id="translateY-value">0.000</span>
                </div>
                <div class="transform-slider-row">
                    <label>Z:</label>
                    <input type="range" id="translateZ" min="-5.0" max="5.0" step="0.001" value="0">
                    <span class="transform-value" id="translateZ-value">0.000</span>
                </div>
            </div>

            <div class="transform-section">
                <div class="transform-section-title">Rotation (deg)</div>
                <div class="transform-slider-row">
                    <label>X:</label>
                    <input type="range" id="rotateX" min="-180" max="180" step="0.1" value="0">
                    <span class="transform-value" id="rotateX-value">0.0°</span>
                </div>
                <div class="transform-slider-row">
                    <label>Y:</label>
                    <input type="range" id="rotateY" min="-180" max="180" step="0.1" value="0">
                    <span class="transform-value" id="rotateY-value">0.0°</span>
                </div>
                <div class="transform-slider-row">
                    <label>Z:</label>
                    <input type="range" id="rotateZ" min="-180" max="180" step="0.1" value="0">
                    <span class="transform-value" id="rotateZ-value">0.0°</span>
                </div>
            </div>

            <div class="transform-section">
                <div class="transform-section-title">Uniform Scale</div>
                <div class="transform-slider-row">
                    <label>Scale:</label>
                    <input type="range" id="scale" min="0.1" max="10.0" step="0.001" value="1.0">
                    <span class="transform-value" id="scale-value">1.000</span>
                </div>
            </div>

            <button id="resetTransform" class="transform-reset-btn">Reset Transform</button>

            <div id="debugTransformInfo" class="transform-debug-info" style="display: none;">
                <div class="transform-section-title">Debug Info</div>
                <pre id="debugTransformText"></pre>
            </div>
        `;

        this.container.innerHTML = html;

        // Cache element references
        this.enableCheckbox = this.container.querySelector('#enableTransform');
        this.debugCheckbox = this.container.querySelector('#showDebugTransform');
        this.debugDisplay = this.container.querySelector('#debugTransformInfo');
        this.debugText = this.container.querySelector('#debugTransformText');
        this.resetBtn = this.container.querySelector('#resetTransform');

        // Cache slider and display references
        ['translateX', 'translateY', 'translateZ', 'rotateX', 'rotateY', 'rotateZ', 'scale'].forEach(id => {
            this.sliders[id] = this.container.querySelector(`#${id}`);
            this.valueDisplays[id] = this.container.querySelector(`#${id}-value`);
        });
    }

    /**
     * Bind event listeners to controls
     */
    _bindEvents() {
        // Enable/disable transformation
        this.enableCheckbox.addEventListener('change', (e) => {
            this.transformer.setEnabled(e.target.checked);
            this._onTransformChange();
        });

        // Debug mode toggle
        this.debugCheckbox.addEventListener('change', (e) => {
            this.transformer.setDebugMode(e.target.checked);
            this.debugDisplay.style.display = e.target.checked ? 'block' : 'none';
            if (e.target.checked) {
                this.updateDebugDisplay();
            }
        });

        // Translation sliders
        this.sliders.translateX.addEventListener('input', (e) => {
            const value = parseFloat(e.target.value);
            this.valueDisplays.translateX.textContent = value.toFixed(3);
            this.transformer.setTranslation(
                value,
                this.transformer.translation.y,
                this.transformer.translation.z
            );
            this._onTransformChange();
        });

        this.sliders.translateY.addEventListener('input', (e) => {
            const value = parseFloat(e.target.value);
            this.valueDisplays.translateY.textContent = value.toFixed(3);
            this.transformer.setTranslation(
                this.transformer.translation.x,
                value,
                this.transformer.translation.z
            );
            this._onTransformChange();
        });

        this.sliders.translateZ.addEventListener('input', (e) => {
            const value = parseFloat(e.target.value);
            this.valueDisplays.translateZ.textContent = value.toFixed(3);
            this.transformer.setTranslation(
                this.transformer.translation.x,
                this.transformer.translation.y,
                value
            );
            this._onTransformChange();
        });

        // Rotation sliders
        this.sliders.rotateX.addEventListener('input', (e) => {
            const value = parseFloat(e.target.value);
            this.valueDisplays.rotateX.textContent = value.toFixed(1) + '°';
            this.transformer.setRotationDegrees(
                value,
                this.transformer.rotation.y * 180 / Math.PI,
                this.transformer.rotation.z * 180 / Math.PI
            );
            this._onTransformChange();
        });

        this.sliders.rotateY.addEventListener('input', (e) => {
            const value = parseFloat(e.target.value);
            this.valueDisplays.rotateY.textContent = value.toFixed(1) + '°';
            this.transformer.setRotationDegrees(
                this.transformer.rotation.x * 180 / Math.PI,
                value,
                this.transformer.rotation.z * 180 / Math.PI
            );
            this._onTransformChange();
        });

        this.sliders.rotateZ.addEventListener('input', (e) => {
            const value = parseFloat(e.target.value);
            this.valueDisplays.rotateZ.textContent = value.toFixed(1) + '°';
            this.transformer.setRotationDegrees(
                this.transformer.rotation.x * 180 / Math.PI,
                this.transformer.rotation.y * 180 / Math.PI,
                value
            );
            this._onTransformChange();
        });

        // Scale slider
        this.sliders.scale.addEventListener('input', (e) => {
            const value = parseFloat(e.target.value);
            this.valueDisplays.scale.textContent = value.toFixed(3);
            this.transformer.setScale(value);
            this._onTransformChange();
        });

        // Reset button
        this.resetBtn.addEventListener('click', () => {
            this.transformer.reset();
            this._updateUIFromTransformer();
            this._onTransformChange();
        });
    }

    /**
     * Update UI values from transformer state
     */
    _updateUIFromTransformer() {
        // Translation
        this.sliders.translateX.value = this.transformer.translation.x;
        this.sliders.translateY.value = this.transformer.translation.y;
        this.sliders.translateZ.value = this.transformer.translation.z;
        this.valueDisplays.translateX.textContent = this.transformer.translation.x.toFixed(3);
        this.valueDisplays.translateY.textContent = this.transformer.translation.y.toFixed(3);
        this.valueDisplays.translateZ.textContent = this.transformer.translation.z.toFixed(3);

        // Rotation (convert radians to degrees for display)
        const rotX = this.transformer.rotation.x * 180 / Math.PI;
        const rotY = this.transformer.rotation.y * 180 / Math.PI;
        const rotZ = this.transformer.rotation.z * 180 / Math.PI;
        this.sliders.rotateX.value = rotX;
        this.sliders.rotateY.value = rotY;
        this.sliders.rotateZ.value = rotZ;
        this.valueDisplays.rotateX.textContent = rotX.toFixed(1) + '°';
        this.valueDisplays.rotateY.textContent = rotY.toFixed(1) + '°';
        this.valueDisplays.rotateZ.textContent = rotZ.toFixed(1) + '°';

        // Scale
        this.sliders.scale.value = this.transformer.scale;
        this.valueDisplays.scale.textContent = this.transformer.scale.toFixed(3);
    }

    /**
     * Called when transformation changes
     */
    _onTransformChange() {
        if (this.transformer.debugMode) {
            this.updateDebugDisplay();
        }

        // Dispatch custom event that viewers can listen to
        const event = new CustomEvent('transform-changed', {
            detail: {
                transformer: this.transformer
            }
        });
        this.container.dispatchEvent(event);
    }

    /**
     * Update debug display with current transformation info
     */
    updateDebugDisplay() {
        if (this.debugText) {
            this.debugText.textContent = this.transformer.getDebugInfo();
        }
    }

    /**
     * Destroy UI and remove event listeners
     */
    destroy() {
        this.container.innerHTML = '';
    }
}
