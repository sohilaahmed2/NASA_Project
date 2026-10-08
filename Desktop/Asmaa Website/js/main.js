/**
 * CAPTAIN ASMAA - Interactive Website Engine
 * Pure Vanilla JavaScript (No heavy dependencies)
 */

document.addEventListener('DOMContentLoaded', () => {
    // 1. NAVBAR SCROLL EFFECT & ACTIVE STATE
    const navbar = document.getElementById('navbar');
    const navLinks = document.querySelectorAll('.nav-link, .mobile-nav-link');
    const sections = document.querySelectorAll('section[id]');
    const scrollTopBtn = document.getElementById('scrollTopBtn');

    window.addEventListener('scroll', () => {
        const scrollY = window.pageYOffset;

        // Navbar background on scroll
        if (scrollY > 50) {
            navbar.classList.add('scrolled');
        } else {
            navbar.classList.remove('scrolled');
        }

        // Scroll to top button visibility
        if (scrollY > 400) {
            scrollTopBtn.classList.add('visible');
        } else {
            scrollTopBtn.classList.remove('visible');
        }

        // Active link highlight
        sections.forEach(current => {
            const sectionHeight = current.offsetHeight;
            const sectionTop = current.offsetTop - 120;
            const sectionId = current.getAttribute('id');

            if (scrollY > sectionTop && scrollY <= sectionTop + sectionHeight) {
                navLinks.forEach(link => {
                    link.classList.remove('active');
                    if (link.getAttribute('href') === `#${sectionId}`) {
                        link.classList.add('active');
                    }
                });
            }
        });
    });

    // Scroll to top action
    if (scrollTopBtn) {
        scrollTopBtn.addEventListener('click', () => {
            window.scrollTo({
                top: 0,
                behavior: 'smooth'
            });
        });
    }

    // 2. MOBILE MENU TOGGLE
    const menuToggle = document.getElementById('menuToggle');
    const mobileMenu = document.getElementById('mobileMenu');

    if (menuToggle && mobileMenu) {
        menuToggle.addEventListener('click', () => {
            const isOpen = mobileMenu.classList.toggle('open');
            menuToggle.innerHTML = isOpen 
                ? '<i class="fas fa-times"></i>' 
                : '<i class="fas fa-bars"></i>';
        });

        // Close mobile menu when link is clicked
        const mobileLinks = mobileMenu.querySelectorAll('a');
        mobileLinks.forEach(link => {
            link.addEventListener('click', () => {
                mobileMenu.classList.remove('open');
                menuToggle.innerHTML = '<i class="fas fa-bars"></i>';
            });
        });
    }

    // 3. REVIEWS SLIDER CONTROLS
    const reviewsSlider = document.getElementById('reviewsSlider');
    const prevReviewBtn = document.getElementById('prevReviewBtn');
    const nextReviewBtn = document.getElementById('nextReviewBtn');

    if (reviewsSlider && prevReviewBtn && nextReviewBtn) {
        prevReviewBtn.addEventListener('click', () => {
            // Note: In RTL, scrollBy positive or negative depends on browser RTL implementation
            // Using scrollBy with 340px width per card
            reviewsSlider.scrollBy({ left: 340, behavior: 'smooth' });
        });

        nextReviewBtn.addEventListener('click', () => {
            reviewsSlider.scrollBy({ left: -340, behavior: 'smooth' });
        });
    }

    // 4. LIGHTBOX MODAL SYSTEM
    const lightboxModal = document.getElementById('lightboxModal');
    const lightboxImg = document.getElementById('lightboxImg');
    const lightboxClose = document.getElementById('lightboxClose');
    const lightboxPrev = document.getElementById('lightboxPrev');
    const lightboxNext = document.getElementById('lightboxNext');

    let currentImages = [];
    let currentIndex = 0;

    function openLightbox(imagesList, index) {
        currentImages = imagesList;
        currentIndex = index;
        lightboxImg.src = currentImages[currentIndex];
        lightboxModal.classList.add('active');
        document.body.style.overflow = 'hidden';
    }

    function closeLightbox() {
        lightboxModal.classList.remove('active');
        document.body.style.overflow = '';
    }

    function showNextImage() {
        if (currentImages.length <= 1) return;
        currentIndex = (currentIndex + 1) % currentImages.length;
        lightboxImg.src = currentImages[currentIndex];
    }

    function showPrevImage() {
        if (currentImages.length <= 1) return;
        currentIndex = (currentIndex - 1 + currentImages.length) % currentImages.length;
        lightboxImg.src = currentImages[currentIndex];
    }

    if (lightboxModal) {
        lightboxClose.addEventListener('click', closeLightbox);
        lightboxNext.addEventListener('click', showNextImage);
        lightboxPrev.addEventListener('click', showPrevImage);

        // Close on background click
        lightboxModal.addEventListener('click', (e) => {
            if (e.target === lightboxModal) {
                closeLightbox();
            }
        });

        // Keyboard controls
        document.addEventListener('keydown', (e) => {
            if (!lightboxModal.classList.contains('active')) return;
            if (e.key === 'Escape') closeLightbox();
            if (e.key === 'ArrowRight') showPrevImage(); // in RTL right is previous
            if (e.key === 'ArrowLeft') showNextImage();  // in RTL left is next
        });
    }

    // Attach Lightbox to Gallery Items
    const galleryItems = document.querySelectorAll('.gallery-item');
    const gallerySrcs = Array.from(galleryItems).map(item => item.getAttribute('data-src') || item.querySelector('img').src);

    galleryItems.forEach((item, idx) => {
        item.addEventListener('click', () => {
            openLightbox(gallerySrcs, idx);
        });
    });

    // Attach Lightbox to Reviews
    const reviewCards = document.querySelectorAll('.review-card');
    const reviewSrcs = Array.from(reviewCards).map(card => card.getAttribute('data-src') || card.querySelector('img').src);

    reviewCards.forEach((card, idx) => {
        card.addEventListener('click', () => {
            openLightbox(reviewSrcs, idx);
        });
    });

    // Attach Lightbox to Moments
    const momentCards = document.querySelectorAll('.moment-card');
    const momentSrcs = Array.from(momentCards).map(card => card.getAttribute('data-src') || card.querySelector('img').src);

    momentCards.forEach((card, idx) => {
        card.addEventListener('click', () => {
            openLightbox(momentSrcs, idx);
        });
    });

    // 5. INTERACTIVE WHATSAPP QUICK INQUIRY BUILDER
    const bookingForm = document.getElementById('quickBookingForm');
    if (bookingForm) {
        bookingForm.addEventListener('submit', (e) => {
            e.preventDefault();
            
            const name = document.getElementById('traineeName').value.trim();
            const goal = document.getElementById('traineeGoal').value;
            const location = document.getElementById('traineeLocation').value;
            const timePref = document.getElementById('traineeTime').value;

            let msg = `مرحباً كابتن أسماء! 👋\nحابب/ة استفسر عن تفاصيل الاشتراك ونظام التدريب معاكي.`;
            if (name) {
                msg += `\nالاسم: ${name}`;
            }
            if (goal) {
                msg += `\nالهدف الرياضي: ${goal}`;
            }
            if (location) {
                msg += `\nالمكان المفضل للتمرين: ${location}`;
            }
            if (timePref) {
                msg += `\nالموعد المناسب: ${timePref}`;
            }

            const encodedMsg = encodeURIComponent(msg);
            const waUrl = `https://wa.me/201033241658?text=${encodedMsg}`;
            
            window.open(waUrl, '_blank');
        });
    }

    // 6. HERO ANIMATED SLIDESHOW ENGINE
    const heroSlides = document.querySelectorAll('.hero-slide');
    const heroDots = document.querySelectorAll('.hero-dot');
    let currentHeroIndex = 0;
    let heroInterval = null;

    function showHeroSlide(index) {
        if (!heroSlides.length) return;
        currentHeroIndex = (index + heroSlides.length) % heroSlides.length;

        heroSlides.forEach((slide, idx) => {
            if (idx === currentHeroIndex) {
                slide.classList.add('active');
            } else {
                slide.classList.remove('active');
            }
        });

        heroDots.forEach((dot, idx) => {
            if (idx === currentHeroIndex) {
                dot.classList.add('active');
            } else {
                dot.classList.remove('active');
            }
        });
    }

    function startHeroSlideshow() {
        if (heroSlides.length <= 1) return;
        if (heroInterval) clearInterval(heroInterval);
        heroInterval = setInterval(() => {
            showHeroSlide(currentHeroIndex + 1);
        }, 3600);
    }

    if (heroSlides.length > 0) {
        startHeroSlideshow();

        heroDots.forEach(dot => {
            dot.addEventListener('click', () => {
                const targetIdx = parseInt(dot.getAttribute('data-index'), 10);
                showHeroSlide(targetIdx);
                startHeroSlideshow(); // reset timer on user click
            });
        });
    }

    // 7. SMOOTH SCROLLING FOR ALL ANCHORS
    document.querySelectorAll('a[href^="#"]').forEach(anchor => {
        anchor.addEventListener('click', function (e) {
            const targetId = this.getAttribute('href');
            if (targetId === '#') return;
            
            const targetElem = document.querySelector(targetId);
            if (targetElem) {
                e.preventDefault();
                const offsetTop = targetElem.offsetTop - 70;
                window.scrollTo({
                    top: offsetTop,
                    behavior: 'smooth'
                });
            }
        });
    });
});
