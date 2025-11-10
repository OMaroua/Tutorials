/**
 * VAE Tutorial - Interactive Components
 * ======================================
 * 
 * This script adds interactive features to the VAE tutorial website.
 * 
 * Author: Maroua Oukrid
 * Date: November 2024
 */

// ============================================
// Initialization
// ============================================

document.addEventListener('DOMContentLoaded', function() {
    console.log('VAE Tutorial - Interactive components loaded');
    
    initSyntaxHighlighting();
    initSmoothScrolling();
    initNavbarEffects();
    initScrollAnimations();
    initInteractiveVisualizations();
    initCopyCodeButtons();
    initProgressIndicator();
});


// ============================================
// Syntax Highlighting
// ============================================

function initSyntaxHighlighting() {
    if (typeof hljs !== 'undefined') {
        hljs.highlightAll();
        console.log('✓ Syntax highlighting initialized');
    }
}


// ============================================
// Smooth Scrolling
// ============================================

function initSmoothScrolling() {
    document.querySelectorAll('a[href^="#"]').forEach(anchor => {
        anchor.addEventListener('click', function(e) {
            e.preventDefault();
            
            const targetId = this.getAttribute('href');
            const targetElement = document.querySelector(targetId);
            
            if (targetElement) {
                const navbarHeight = document.querySelector('.navbar').offsetHeight;
                const targetPosition = targetElement.getBoundingClientRect().top + window.pageYOffset - navbarHeight - 20;
                
                window.scrollTo({
                    top: targetPosition,
                    behavior: 'smooth'
                });
                
                // Update active nav link
                updateActiveNavLink(targetId);
            }
        });
    });
    
    console.log('✓ Smooth scrolling initialized');
}


function updateActiveNavLink(targetId) {
    document.querySelectorAll('.nav-menu a').forEach(link => {
        link.classList.remove('active');
    });
    
    const activeLink = document.querySelector(`.nav-menu a[href="${targetId}"]`);
    if (activeLink) {
        activeLink.classList.add('active');
    }
}


// ============================================
// Navbar Scroll Effects
// ============================================

function initNavbarEffects() {
    let lastScroll = 0;
    const navbar = document.querySelector('.navbar');
    
    window.addEventListener('scroll', () => {
        const currentScroll = window.pageYOffset;
        
        // Add shadow when scrolled
        if (currentScroll > 100) {
            navbar.classList.add('scrolled');
        } else {
            navbar.classList.remove('scrolled');
        }
        
        lastScroll = currentScroll;
    });
    
    console.log('✓ Navbar effects initialized');
}


// ============================================
// Scroll Animations
// ============================================

function initScrollAnimations() {
    const observerOptions = {
        threshold: 0.1,
        rootMargin: '0px 0px -50px 0px'
    };
    
    const observer = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                entry.target.classList.add('fade-in');
                
                // Add stagger effect for list items
                if (entry.target.classList.contains('content-card')) {
                    animateListItems(entry.target);
                }
            }
        });
    }, observerOptions);
    
    // Observe content cards and images
    document.querySelectorAll('.content-card, .image-container, .component-card').forEach(el => {
        observer.observe(el);
    });
    
    console.log('✓ Scroll animations initialized');
}


function animateListItems(container) {
    const listItems = container.querySelectorAll('li');
    listItems.forEach((item, index) => {
        setTimeout(() => {
            item.style.opacity = '0';
            item.style.transform = 'translateX(-20px)';
            item.style.transition = 'all 0.4s ease';
            
            setTimeout(() => {
                item.style.opacity = '1';
                item.style.transform = 'translateX(0)';
            }, 50);
        }, index * 50);
    });
}


// ============================================
// Interactive Visualizations
// ============================================

function initInteractiveVisualizations() {
    // Add hover effects to metric cards
    const metricCards = document.querySelectorAll('.metric-card');
    metricCards.forEach(card => {
        card.addEventListener('mouseenter', function() {
            this.style.transform = 'translateY(-8px) scale(1.05)';
        });
        
        card.addEventListener('mouseleave', function() {
            this.style.transform = 'translateY(0) scale(1)';
        });
    });
    
    // Add click-to-zoom for images
    const images = document.querySelectorAll('.result-image, .slice-image');
    images.forEach(img => {
        img.style.cursor = 'pointer';
        img.addEventListener('click', function() {
            openImageModal(this.src, this.alt);
        });
    });
    
    // Interactive latent space demo (if canvas exists)
    const latentCanvas = document.getElementById('latent-canvas');
    if (latentCanvas) {
        initLatentSpaceDemo(latentCanvas);
    }
    
    console.log('✓ Interactive visualizations initialized');
}


function openImageModal(src, alt) {
    // Create modal
    const modal = document.createElement('div');
    modal.className = 'image-modal';
    modal.innerHTML = `
        <div class="modal-backdrop"></div>
        <div class="modal-content">
            <img src="${src}" alt="${alt}">
            <p class="modal-caption">${alt}</p>
            <button class="modal-close">&times;</button>
        </div>
    `;
    
    // Add styles
    modal.style.cssText = `
        position: fixed;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        z-index: 10000;
        display: flex;
        align-items: center;
        justify-content: center;
    `;
    
    const backdrop = modal.querySelector('.modal-backdrop');
    backdrop.style.cssText = `
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        background: rgba(0, 0, 0, 0.9);
    `;
    
    const content = modal.querySelector('.modal-content');
    content.style.cssText = `
        position: relative;
        max-width: 90%;
        max-height: 90%;
        z-index: 1;
    `;
    
    const img = modal.querySelector('img');
    img.style.cssText = `
        max-width: 100%;
        max-height: 85vh;
        border-radius: 8px;
        box-shadow: 0 20px 60px rgba(0, 0, 0, 0.5);
    `;
    
    const caption = modal.querySelector('.modal-caption');
    caption.style.cssText = `
        color: white;
        text-align: center;
        margin-top: 1rem;
        font-size: 1rem;
    `;
    
    const closeBtn = modal.querySelector('.modal-close');
    closeBtn.style.cssText = `
        position: absolute;
        top: -40px;
        right: 0;
        background: none;
        border: none;
        color: white;
        font-size: 3rem;
        cursor: pointer;
        line-height: 1;
    `;
    
    // Close handlers
    const closeModal = () => modal.remove();
    backdrop.addEventListener('click', closeModal);
    closeBtn.addEventListener('click', closeModal);
    document.addEventListener('keydown', function(e) {
        if (e.key === 'Escape') closeModal();
    });
    
    document.body.appendChild(modal);
}


// ============================================
// Copy Code Buttons
// ============================================

function initCopyCodeButtons() {
    const codeBlocks = document.querySelectorAll('.code-block pre');
    
    codeBlocks.forEach(block => {
        const button = document.createElement('button');
        button.className = 'copy-code-btn';
        button.textContent = 'Copy';
        button.style.cssText = `
            position: absolute;
            top: 8px;
            right: 8px;
            background: rgba(99, 102, 241, 0.8);
            color: white;
            border: none;
            padding: 6px 12px;
            border-radius: 4px;
            cursor: pointer;
            font-size: 0.875rem;
            opacity: 0;
            transition: opacity 0.3s;
        `;
        
        // Make parent relative
        block.parentElement.style.position = 'relative';
        
        // Show button on hover
        block.parentElement.addEventListener('mouseenter', () => {
            button.style.opacity = '1';
        });
        
        block.parentElement.addEventListener('mouseleave', () => {
            button.style.opacity = '0';
        });
        
        // Copy functionality
        button.addEventListener('click', async () => {
            const code = block.querySelector('code').textContent;
            
            try {
                await navigator.clipboard.writeText(code);
                button.textContent = 'Copied!';
                button.style.background = 'rgba(16, 185, 129, 0.8)';
                
                setTimeout(() => {
                    button.textContent = 'Copy';
                    button.style.background = 'rgba(99, 102, 241, 0.8)';
                }, 2000);
            } catch (err) {
                console.error('Failed to copy:', err);
                button.textContent = 'Failed';
                setTimeout(() => {
                    button.textContent = 'Copy';
                }, 2000);
            }
        });
        
        block.parentElement.appendChild(button);
    });
    
    console.log('✓ Copy code buttons initialized');
}


// ============================================
// Progress Indicator
// ============================================

function initProgressIndicator() {
    // Create progress bar
    const progressBar = document.createElement('div');
    progressBar.className = 'reading-progress';
    progressBar.style.cssText = `
        position: fixed;
        top: 0;
        left: 0;
        height: 3px;
        background: linear-gradient(90deg, #6366f1, #ec4899);
        z-index: 9999;
        transition: width 0.1s ease;
    `;
    
    document.body.appendChild(progressBar);
    
    // Update progress on scroll
    window.addEventListener('scroll', () => {
        const windowHeight = window.innerHeight;
        const documentHeight = document.documentElement.scrollHeight - windowHeight;
        const scrolled = window.pageYOffset;
        const progress = (scrolled / documentHeight) * 100;
        
        progressBar.style.width = `${progress}%`;
    });
    
    console.log('✓ Progress indicator initialized');
}


// ============================================
// Interactive Latent Space Demo (Optional)
// ============================================

function initLatentSpaceDemo(canvas) {
    const ctx = canvas.getContext('2d');
    const width = canvas.width = 400;
    const height = canvas.height = 400;
    
    // Generate sample points
    const points = [];
    const numPoints = 1000;
    
    for (let i = 0; i < numPoints; i++) {
        const theta = Math.random() * 2 * Math.PI;
        const r = Math.random() * 150 + 50;
        const x = width / 2 + r * Math.cos(theta);
        const y = height / 2 + r * Math.sin(theta);
        const color = Math.floor(Math.random() * 10);
        
        points.push({ x, y, color });
    }
    
    // Color palette
    const colors = [
        '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
        '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf'
    ];
    
    // Draw points
    function draw() {
        ctx.clearRect(0, 0, width, height);
        
        // Draw grid
        ctx.strokeStyle = 'rgba(255, 255, 255, 0.1)';
        ctx.lineWidth = 1;
        
        for (let i = 0; i <= width; i += 40) {
            ctx.beginPath();
            ctx.moveTo(i, 0);
            ctx.lineTo(i, height);
            ctx.stroke();
            
            ctx.beginPath();
            ctx.moveTo(0, i);
            ctx.lineTo(width, i);
            ctx.stroke();
        }
        
        // Draw points
        points.forEach(point => {
            ctx.fillStyle = colors[point.color];
            ctx.beginPath();
            ctx.arc(point.x, point.y, 3, 0, 2 * Math.PI);
            ctx.fill();
        });
    }
    
    // Interactive hover
    canvas.addEventListener('mousemove', (e) => {
        const rect = canvas.getBoundingClientRect();
        const mouseX = e.clientX - rect.left;
        const mouseY = e.clientY - rect.top;
        
        // Highlight nearby points
        ctx.clearRect(0, 0, width, height);
        draw();
        
        points.forEach(point => {
            const dist = Math.sqrt(
                Math.pow(point.x - mouseX, 2) + 
                Math.pow(point.y - mouseY, 2)
            );
            
            if (dist < 30) {
                ctx.strokeStyle = colors[point.color];
                ctx.lineWidth = 2;
                ctx.beginPath();
                ctx.arc(point.x, point.y, 8, 0, 2 * Math.PI);
                ctx.stroke();
            }
        });
    });
    
    draw();
    console.log('✓ Interactive latent space demo initialized');
}


// ============================================
// Utility Functions
// ============================================

// Throttle function for performance
function throttle(func, wait) {
    let timeout;
    return function executedFunction(...args) {
        const later = () => {
            clearTimeout(timeout);
            func(...args);
        };
        clearTimeout(timeout);
        timeout = setTimeout(later, wait);
    };
}


// Check if element is in viewport
function isInViewport(element) {
    const rect = element.getBoundingClientRect();
    return (
        rect.top >= 0 &&
        rect.left >= 0 &&
        rect.bottom <= (window.innerHeight || document.documentElement.clientHeight) &&
        rect.right <= (window.innerWidth || document.documentElement.clientWidth)
    );
}


// ============================================
// Export for testing
// ============================================

if (typeof module !== 'undefined' && module.exports) {
    module.exports = {
        initSmoothScrolling,
        initNavbarEffects,
        initScrollAnimations,
        throttle,
        isInViewport
    };
}

console.log('✓ VAE Tutorial JavaScript loaded successfully');

